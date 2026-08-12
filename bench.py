"""Recall@k vs latency across a binary-quantization oversampling sweep.
One dataset, one collection, both numbers recorded per configuration.

Ground truth = exact KNN on the same collection (the documented ANN-recall method:
qdrant.tech/documentation/tutorials-search-engineering/ann-recall/#automate-in-ci-with-python).

usage: python bench.py ingest
       python bench.py bench [k]     # k defaults to 10
"""
import csv, glob, json, statistics, sys, time

import numpy as np
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

COLL = "bq_bench"
DIM = 1536
N_INDEX = 100_000
N_QUERY = 1_000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3  # latency percentiles are medianed over these

# 3 shards x 38,462 rows covers N_INDEX + N_QUERY with room to spare
REPO = "KShivendu/dbpedia-entities-openai-1M"
SHARDS = [
    "data/train-00000-of-00026-3c7b99d1c7eda36e.parquet",
    "data/train-00001-of-00026-2b24035a6390fdcb.parquet",
    "data/train-00002-of-00026-b05ce48965853dad.parquet",
]

client = QdrantClient(host="localhost", prefer_grpc=True, timeout=600)


def download():
    from huggingface_hub import hf_hub_download
    for f in SHARDS:
        print(hf_hub_download(REPO, f, repo_type="dataset", local_dir="data"), flush=True)


def load_vectors():
    if not glob.glob("data/data/*.parquet"):
        download()
    vecs = []
    for f in sorted(glob.glob("data/data/*.parquet")):
        vecs.append(np.stack(pq.read_table(f, columns=["openai"])["openai"].to_numpy(zero_copy_only=False)).astype(np.float32))
        if sum(len(v) for v in vecs) >= N_INDEX + N_QUERY:
            break
    return np.concatenate(vecs)


def ingest():
    all_v = load_vectors()
    assert len(all_v) >= N_INDEX + N_QUERY, len(all_v)
    corpus, queries = all_v[:N_INDEX], all_v[N_INDEX:N_INDEX + N_QUERY]
    np.save("queries.npy", queries)

    client.delete_collection(COLL)
    client.create_collection(
        COLL,
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
        # ponytail: originals stay in RAM so rescore latency measures quantization, not disk IO
        quantization_config=models.BinaryQuantization(
            binary=models.BinaryQuantizationConfig(always_ram=True)
        ),
    )
    for i in range(0, N_INDEX, 2000):
        batch = corpus[i:i + 2000]
        client.upsert(
            COLL,
            points=models.Batch(ids=list(range(i, i + len(batch))), vectors=batch.tolist()),
            wait=False,
        )
        print(f"upserted {i + len(batch)}", flush=True)

    while True:
        info = client.get_collection(COLL)
        print(info.status, "indexed:", info.indexed_vectors_count, flush=True)
        if info.status == models.CollectionStatus.GREEN and info.indexed_vectors_count >= N_INDEX:
            break
        time.sleep(10)


NO_QUANT = models.QuantizationSearchParams(ignore=True)
# The candidate pool oversampling asks for: "if oversampling is 2.4 and limit is 100,
# then 240 vectors will be pre-selected" (qdrant.tech/documentation/manage-data/quantization).
pool = lambda o: max(K, round(K * o))
OVERSAMPLING = (1, 1.5, 2, 3, 4, 5, 6, 8, 12, 16)
EF_SWEEP = (100, 128, 200, 300, 512)  # 100 == ef_construct, to identify the unset default

# Oversampling raises the candidate count the index has to walk, so a BQ run at 16x
# is not comparable to an unquantized run at the engine's default ef. Both sides get
# an ef sweep, and ef=300 is repeated across the whole oversampling range so one
# comparison holds ef fixed.
CONFIGS = (
    [("exact (ground truth)", dict(exact=True, quantization=NO_QUANT))]
    + [("HNSW, no quantization", dict(quantization=NO_QUANT))]
    + [(f"HNSW, no quantization, ef={ef}", dict(quantization=NO_QUANT, hnsw_ef=ef)) for ef in EF_SWEEP]
    # rescore off across the same oversampling range: does oversampling alone recover anything?
    + [(f"BQ, no rescore, oversampling {o}x",
        dict(quantization=models.QuantizationSearchParams(rescore=False, oversampling=float(o))))
       for o in OVERSAMPLING]
    # and the same with ef pinned, so no-quant / quant-without-rescore / quant-with-rescore
    # can be compared at one shared ef
    + [(f"BQ, no rescore, oversampling {o}x, ef=300",
        dict(quantization=models.QuantizationSearchParams(rescore=False, oversampling=float(o)), hnsw_ef=300))
       for o in OVERSAMPLING]
    + [(f"BQ + rescore, oversampling {o}x",
        dict(quantization=models.QuantizationSearchParams(rescore=True, oversampling=float(o))))
       for o in OVERSAMPLING]
    + [(f"BQ + rescore, oversampling {o}x, ef=300",
        dict(quantization=models.QuantizationSearchParams(rescore=True, oversampling=float(o)), hnsw_ef=300))
       for o in OVERSAMPLING]
    # The paired control. Oversampling pre-selects limit*oversampling candidates, so the
    # matched unquantized run is one that walks the same size pool: ef = limit*oversampling.
    # Then the only difference left is 1-bit vs float distance computation.
    + [(f"HNSW, no quantization, ef=pool({o}x)",
        dict(quantization=NO_QUANT, hnsw_ef=pool(o)))
       for o in OVERSAMPLING]
    # Same pool, pinned on the quantized side too, so ef cannot silently bind instead.
    + [(f"BQ + rescore, oversampling {o}x, ef=pool",
        dict(quantization=models.QuantizationSearchParams(rescore=True, oversampling=float(o)), hnsw_ef=pool(o)))
       for o in OVERSAMPLING]
)


def run(queries, params):
    ids, lat = [], []
    for q in queries:
        t = time.perf_counter()
        r = client.query_points(COLL, query=q.tolist(), limit=K,
                                search_params=models.SearchParams(**params)).points
        lat.append((time.perf_counter() - t) * 1000)
        ids.append({p.id for p in r})
    return ids, lat


def recall_at_k(ids, truth, k):
    """Mean overlap between approximate and exact id sets, divided by k."""
    return statistics.fmean(len(a & b) / k for a, b in zip(ids, truth))


def pct(sorted_lat, p):
    return sorted_lat[int(p * len(sorted_lat))]


def bench():
    queries = np.load("queries.npy")
    print("warmup", flush=True)
    run(queries[:200], CONFIGS[1][1])

    # Passes are INTERLEAVED: every configuration is measured once per round, rather than
    # all its passes back-to-back. Nesting passes inside a configuration lets one contended
    # window land entirely on one point - it once put two measurements of an identical
    # search 2x apart. Round-robin spreads any drift across the whole curve instead.
    passes = {label: [] for label, _ in CONFIGS}
    first_ids = {}
    for rnd in range(REPS):
        for label, params in CONFIGS:
            ids, lat = run(queries, params)
            lat.sort()
            passes[label].append(lat)
            first_ids.setdefault(label, ids)
        print(f"round {rnd + 1}/{REPS} done", flush=True)

    rows = []
    truth = None
    for label, params in CONFIGS:
        # Recall is deterministic, so one pass settles it; latency takes the median of rounds.
        ids = first_ids[label]
        reps = passes[label]
        if truth is None:
            truth = ids
            recall = 1.0
        else:
            recall = recall_at_k(ids, truth, K)
        p95s = [pct(l, 0.95) for l in reps]
        p50s = [statistics.median(l) for l in reps]
        q = params.get("quantization")
        row = dict(config=label, limit=K,
                   oversampling=q.oversampling if q else None,
                   rescore=q.rescore if q else None,
                   hnsw_ef=params.get("hnsw_ef"),
                   exact=bool(params.get("exact")),
                   recall=round(recall, 4),
                   p50_ms=round(statistics.median(p50s), 3),
                   p95_ms=round(statistics.median(p95s), 3),
                   # Contention is one-sided: it only ever adds time. The fastest pass is
                   # therefore the best estimate of the engine's own cost, and the median
                   # still carries whatever contention every pass happened to share.
                   p50_min_ms=round(min(p50s), 3),
                   p95_min_ms=round(min(p95s), 3),
                   p95_max_ms=round(max(p95s), 3),
                   p99_ms=round(statistics.median([pct(l, 0.99) for l in reps]), 3),
                   mean_ms=round(statistics.median([statistics.fmean(l) for l in reps]), 3),
                   p95_spread_ms=round(max(p95s) - min(p95s), 3),
                   reps=REPS)
        rows.append(row)
        print(json.dumps(row), flush=True)

    json.dump(rows, open(f"results_k{K}.json", "w"), indent=2)
    write_csv(rows, f"results_k{K}.csv")


FIELDS = ["config", "limit", "oversampling", "rescore", "hnsw_ef", "exact",
          "recall", "p50_ms", "p95_ms", "p99_ms", "mean_ms",
          "p50_min_ms", "p95_min_ms", "p95_max_ms", "p95_spread_ms", "reps"]


def write_csv(rows, path):
    # config labels contain commas, so quote properly instead of hand-joining
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    {"ingest": ingest, "bench": bench}[sys.argv[1]]()

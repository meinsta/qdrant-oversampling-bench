"""Recall@k vs latency across a binary-quantization oversampling sweep.
One dataset, one collection, both numbers recorded per configuration.

Ground truth = exact KNN on the same collection (the documented ANN-recall method:
qdrant.tech/documentation/tutorials-search-engineering/ann-recall/#automate-in-ci-with-python).

usage: python bench.py ingest
       python bench.py bench [k]     # k defaults to 10
"""
import glob, json, statistics, sys, time

import numpy as np
import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

COLL = "bq_bench"
DIM = 1536
N_INDEX = 100_000
N_QUERY = 1_000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 10

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


CONFIGS = [
    ("exact (ground truth)", dict(exact=True, quantization=models.QuantizationSearchParams(ignore=True))),
    ("HNSW, no quantization", dict(quantization=models.QuantizationSearchParams(ignore=True))),
    ("BQ, no rescore", dict(quantization=models.QuantizationSearchParams(rescore=False, oversampling=1.0))),
] + [
    (f"BQ + rescore, oversampling {o}x",
     dict(quantization=models.QuantizationSearchParams(rescore=True, oversampling=float(o))))
    for o in (1, 1.5, 2, 3, 4, 5, 6, 8, 12, 16)
]


def run(queries, params):
    ids, lat = [], []
    for q in queries:
        t = time.perf_counter()
        r = client.query_points(COLL, query=q.tolist(), limit=K,
                                search_params=models.SearchParams(**params)).points
        lat.append((time.perf_counter() - t) * 1000)
        ids.append({p.id for p in r})
    return ids, lat


def bench():
    queries = np.load("queries.npy")
    print("warmup", flush=True)
    run(queries[:50], CONFIGS[1][1])

    rows = []
    truth = None
    for label, params in CONFIGS:
        ids, lat = run(queries, params)
        if truth is None:
            truth = ids
            recall = 1.0
        else:
            recall = statistics.fmean(len(a & b) / K for a, b in zip(ids, truth))
        lat.sort()
        row = dict(config=label, recall=round(recall, 4),
                   p50_ms=round(statistics.median(lat), 3),
                   p95_ms=round(lat[int(0.95 * len(lat))], 3),
                   p99_ms=round(lat[int(0.99 * len(lat))], 3),
                   mean_ms=round(statistics.fmean(lat), 3),
                   oversampling=params.get("quantization").oversampling if params.get("quantization") else None)
        rows.append(row)
        print(json.dumps(row), flush=True)

    json.dump(rows, open(f"results_k{K}.json", "w"), indent=2)
    with open(f"results_k{K}.csv", "w") as f:
        f.write("config,oversampling,recall,p50_ms,p95_ms,p99_ms,mean_ms\n")
        for r in rows:
            f.write(f"{r['config']},{r['oversampling']},{r['recall']},{r['p50_ms']},{r['p95_ms']},{r['p99_ms']},{r['mean_ms']}\n")


if __name__ == "__main__":
    {"ingest": ingest, "bench": bench}[sys.argv[1]]()

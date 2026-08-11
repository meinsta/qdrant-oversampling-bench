# Oversampling: recall vs latency, measured

Qdrant's documentation says oversampling trades search speed for search quality, and
recommends 3x for binary quantization. Both halves of that trade are documented separately:
the [binary quantization article](https://qdrant.tech/articles/binary-quantization/) plots
recall against candidates, the
[OpenAI embeddings article](https://qdrant.tech/articles/binary-quantization-openai/) reports
recall at 1x/2x/3x with no latency column, and the
[quantization docs](https://qdrant.tech/documentation/manage-data/quantization/) describe the
tradeoff in prose. No published source puts recall and latency on one chart, for one dataset,
across the oversampling range.

This repo is that measurement.

**[Read the report](https://claude.ai/code/artifact/9cad4724-5673-415f-8b2b-95df43633db1)**
(charts, full tables, method) or open `report.html` locally.

## Results

100,000 DBpedia entity vectors, OpenAI `text-embedding-ada-002` (1536d, cosine), 1,000 held-out
queries, binary quantization with vectors in RAM, Qdrant 1.19.0 on an Apple M5.

| Configuration | recall@10 | p95 (ms) | recall@100 | p95 (ms) |
|---|---|---|---|---|
| Exact KNN (ground truth) | 1.0000 | 8.38 | 1.0000 | 8.70 |
| HNSW, no quantization | 0.9943 | 1.87 | 0.9877 | 2.25 |
| BQ, rescore off | 0.6725 | 0.67 | 0.6873 | 1.06 |
| BQ + rescore, 1x | 0.9238 | 0.65 | 0.9496 | 1.14 |
| BQ + rescore, 2x | 0.9747 | 0.68 | 0.9871 | 1.31 |
| **BQ + rescore, 3x** | **0.9862** | **0.75** | **0.9946** | **1.49** |
| BQ + rescore, 4x | 0.9900 | 0.74 | 0.9972 | 1.66 |
| BQ + rescore, 8x | 0.9931 | 0.72 | 0.9992 | 2.19 |
| BQ + rescore, 16x | 0.9966 | 0.87 | 0.9997 | 3.10 |

Full sweep including 1.5x, 5x, 6x, 12x and p50/p99 in `results_k10.csv` and `results_k100.csv`.

### What the numbers say

**The tradeoff is real, but its slope is set by `k`.** At k=10 the whole sweep from 1x to 16x
costs 0.22 ms of p95 and buys 7.3 points of recall, so there is no reason to run it low. At
k=100 the same sweep costs 2.0 ms, and every step past 4x pays real latency for hundredths of
a point.

**At k=100, 3x oversampling with rescore beats un-quantized HNSW on both axes:** 0.9946 recall
at 1.49 ms against 0.9877 at 2.25 ms, while searching a 32x smaller index. The recommended
default lands on the useful part of the curve.

**Rescore is not optional.** With it off, binary quantization returns 0.67 recall@10 and no
oversampling value rescues it, because there is nothing to re-rank with.

## Run it

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# Qdrant 1.19.0 - any local instance works; Docker or the release binary
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.19.0

./.venv/bin/python bench.py ingest       # downloads ~1 GB, indexes 100k, waits for green
./.venv/bin/python bench.py bench 10     # -> results_k10.csv
./.venv/bin/python bench.py bench 100    # -> results_k100.csv
```

Add a configuration by appending to `CONFIGS` in `bench.py`. Scalar quantization is a one-line
change there. `python test_bench.py` checks the recall math and that the published CSVs still
agree with the JSON.

## Method

- **Ground truth.** Exact KNN over the same collection (`exact=true`, quantization ignored),
  following Qdrant's
  [ANN-recall method](https://qdrant.tech/documentation/tutorials-search-engineering/ann-recall/#automate-in-ci-with-python).
  Recall is the overlap of the approximate and exact id sets divided by k, averaged over 1,000
  queries. Every golden answer is reachable by construction, so this isolates index and
  quantization loss. It says nothing about whether the embedding model retrieves the right thing.
- **Index.** Default HNSW (m=16, ef_construct=100, default query ef), binary quantization with
  `always_ram`, original vectors in RAM. The collection is green and fully indexed before any
  query is timed.
- **Timing.** 1,000 sequential single queries per configuration over local gRPC, 50-query warmup,
  client-side wall clock.

## What these numbers do not cover

- **Transport is included.** Roughly 0.3 ms of every figure is client and gRPC overhead. Read the
  distances between configurations, not the absolute values.
- **Rescoring from disk.** Originals are in RAM here. The docs warn that disk-backed rescoring is
  much slower, and this run does not measure it.
- **Concurrency.** Sequential queries only. Throughput under load is a separate measurement.
- **One model, one dimensionality.** Binary quantization is sensitive to how a model distributes
  its components, and 1536d OpenAI vectors are close to its best case. Re-run before quoting
  these numbers for a different model.
- **One node, 100k vectors.** Both curves shift with collection size and shard count.

## License

Apache-2.0. See [LICENSE](LICENSE).

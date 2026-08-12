# Oversampling: recall vs latency, measured

Qdrant's documentation says oversampling trades search speed for search quality, and
recommends 3x for binary quantization. Both halves of that trade are documented separately:
the [binary quantization article](https://qdrant.tech/articles/binary-quantization/) plots
recall against candidates, the
[OpenAI embeddings article](https://qdrant.tech/articles/binary-quantization-openai/) sweeps
oversampling x rescore x limit and reports recall with no latency column, and the
[quantization docs](https://qdrant.tech/documentation/manage-data/quantization/) describe the
tradeoff in prose. No published source puts recall and latency on one chart, for one dataset,
across the oversampling range.

This repo is that measurement. It follows the OpenAI article's grid (oversampling x rescore x
limit, limits 10/20/50/100) and adds the two things needed to read a tradeoff: latency
percentiles, and an `hnsw_ef` sweep on **both** sides so the comparison is honest.

**[Read the report](https://claude.ai/code/artifact/9cad4724-5673-415f-8b2b-95df43633db1)**
(charts, full tables, method) or open `report.html` locally.

## Setup

100,000 DBpedia entity vectors, OpenAI `text-embedding-ada-002` (1536d, cosine), 1,000 held-out
queries, binary quantization with vectors in RAM, Qdrant 1.19.0 on an Apple M5. Ground truth is
exact KNN on the same collection. Every cell below is `recall / p95 ms` over 1,000 sequential
queries.

## Why `hnsw_ef` has to be swept on both sides

Oversampling makes the index walk more candidates before rescoring, so binary quantization at
16x is doing far more graph work than an unquantized search at the engine's default `ef`.
Comparing the two directly flatters quantization. The fix is to sweep `ef` on the unquantized
side too, and to repeat the whole oversampling range at a fixed `ef`.

### Unquantized HNSW, by `ef`

| Configuration | limit 10 | limit 20 | limit 50 | limit 100 |
|---|---|---|---|---|
| ef unset (engine default) | 0.9943 / 1.99 | 0.9940 / 1.99 | 0.9919 / 2.40 | 0.9877 / 2.29 |
| ef=100 | 0.9943 / 1.97 | 0.9940 / 2.15 | 0.9919 / 2.14 | 0.9877 / 2.28 |
| ef=128 | 0.9956 / 2.23 | 0.9955 / 2.41 | 0.9945 / 2.62 | 0.9920 / 2.73 |
| ef=200 | 0.9976 / 2.88 | 0.9975 / 2.93 | 0.9970 / 3.01 | 0.9960 / 3.13 |
| ef=300 | 0.9987 / 3.71 | 0.9986 / 3.57 | 0.9984 / 3.76 | 0.9979 / 4.42 |
| ef=512 | 0.9992 / 4.86 | 0.9992 / 5.20 | 0.9992 / 5.22 | 0.9990 / 5.12 |

**Unset resolves to `ef=100`.** Recall is identical to four decimals at every limit, and 100 is
this collection's `ef_construct`. The docs do not state the default; this is what it measures as.

### Binary quantization + rescore, at `ef=300` (matched to the baseline)

| Oversampling | limit 10 | limit 20 | limit 50 | limit 100 |
|---|---|---|---|---|
| 1x | 0.9277 / 0.98 | 0.9387 / 1.01 | 0.9453 / 1.29 | 0.9532 / 1.28 |
| 2x | 0.9786 / 1.09 | 0.9846 / 0.90 | 0.9857 / 1.07 | 0.9883 / 1.67 |
| 3x | 0.9904 / 1.00 | 0.9920 / 0.89 | 0.9939 / 1.11 | 0.9946 / 1.49 |
| 4x | 0.9940 / 0.99 | 0.9951 / 1.06 | 0.9961 / 1.15 | 0.9972 / 1.74 |
| **5x** | 0.9957 / 1.12 | 0.9966 / 0.99 | 0.9973 / 1.20 | **0.9982 / 1.77** |
| 6x | 0.9967 / 1.10 | 0.9975 / 0.97 | 0.9978 / 1.25 | 0.9988 / 1.92 |
| **8x** | 0.9975 / 1.02 | 0.9982 / 1.02 | **0.9987 / 1.41** | 0.9992 / 2.22 |
| **12x** | **0.9987 / 1.15** | **0.9986 / 1.16** | 0.9994 / 1.73 | 0.9997 / 2.68 |
| 16x | 0.9989 / 1.11 | 0.9988 / 1.22 | 0.9996 / 2.02 | 0.9997 / 3.14 |

Bold is the cheapest oversampling that matches or beats the `ef=300` unquantized baseline on
**both** recall and p95. Full sweep including 1.5x, the default-ef runs, and p50/p99 in
`results_k*.csv`.

## What the numbers say

**At matched `ef`, quantization wins on both axes at every limit — but it takes more
oversampling than 3x to get there.**

| limit | unquantized ef=300 | cheapest BQ that matches or beats it | speedup |
|---|---|---|---|
| 10 | 0.9987 / 3.71 ms | 12x, 0.9987 / 1.15 ms | 3.2x |
| 20 | 0.9986 / 3.57 ms | 12x, 0.9986 / 1.16 ms | 3.1x |
| 50 | 0.9984 / 3.76 ms | 8x, 0.9987 / 1.41 ms | 2.7x |
| 100 | 0.9979 / 4.42 ms | 5x, 0.9982 / 1.77 ms | 2.5x |

**Bigger `limit` needs less oversampling, not more.** At limit 100 the rescore pool is already
500 vectors at 5x, enough to fix the ranking; at limit 10 the same multiplier only buys 50, so
you need 12x to reach the same recall. Tune oversampling against your actual `limit`.

**The recommended 3x is a speed setting, not a parity setting.** It does not match a tuned
unquantized index at any limit here. What it does do is beat the *default-ef* baseline on both
axes at limit 100 (0.9946 / 1.51 ms against 0.9877 / 2.29 ms). If you want parity with a tuned
index, go to 5x-12x depending on limit.

**Oversampling is much cheaper than `ef` for the same recall.** Reading the two frontiers: to
buy 0.9987 recall at limit 10, `ef` costs 3.71 ms and oversampling costs 1.15 ms. Both knobs walk
more candidates; only one of them walks them as 1-bit vectors. Note that this is a latency win,
not a memory win: rescoring keeps the original float vectors resident, so this configuration
stores the 1-bit copy *in addition to* them.

**Rescore is not optional.** With it off, binary quantization returns 0.67-0.69 recall at every
limit, and no oversampling value rescues it, because there is nothing to re-rank with.

## Run it

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# Qdrant 1.19.0 - any local instance works; Docker or the release binary
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.19.0

./.venv/bin/python bench.py ingest                    # downloads ~1 GB, indexes 100k, waits for green
for k in 10 20 50 100; do ./.venv/bin/python bench.py bench $k; done
./.venv/bin/python summarize.py                       # the tables above
```

Add a configuration by appending to `CONFIGS` in `bench.py`. Scalar quantization is a one-line
change there. `python test_bench.py` checks the recall math, the CSV quoting, and that the
published files still agree with each other.

## Method

- **Ground truth.** Exact KNN over the same collection (`exact=true`, quantization ignored),
  following Qdrant's
  [ANN-recall method](https://qdrant.tech/documentation/tutorials-search-engineering/ann-recall/#automate-in-ci-with-python).
  Recall is the overlap of the approximate and exact id sets divided by the limit, averaged over
  1,000 queries. Every golden answer is reachable by construction, so this isolates index and
  quantization loss. It says nothing about whether the embedding model retrieves the right thing.
- **Index.** m=16, ef_construct=100, binary quantization with `always_ram`, original vectors in
  RAM. The collection is green and fully indexed before any query is timed.
- **Timing.** 1,000 sequential single queries per configuration over local gRPC, 200-query warmup,
  client-side wall clock, nothing else running on the host.

## What these numbers do not cover

- **p95 has a noise floor of roughly 0.2-0.3 ms.** A few cells break monotonicity by less than
  that (12x at limit 20, for instance). Differences smaller than 0.3 ms are not signal; the
  frontier-level gaps quoted above are 1-3 ms. `p50_ms` in the CSVs is steadier if you need
  finer resolution.
- **Transport is included.** Roughly 0.3 ms of every figure is client and gRPC overhead. Read the
  distances between configurations, not the absolute values.
- **Rescoring from disk.** Originals are in RAM here. The docs warn that disk-backed rescoring is
  much slower, and this run does not measure it.
- **Concurrency.** Sequential queries only. Throughput under load is a separate measurement.
- **One model, one dimensionality.** Binary quantization is sensitive to how a model distributes
  its components, and 1536d OpenAI vectors are close to its best case. The OpenAI article sweeps
  512-3072 dimensions across two models; this repo does not. Re-run before quoting these numbers
  for a different model.
- **One node, 100k vectors.** Both curves shift with collection size and shard count.

## License

Apache-2.0. See [LICENSE](LICENSE).

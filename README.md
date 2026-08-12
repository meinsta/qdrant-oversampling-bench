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
| ef unset (engine default) | 0.9943 / 2.01 | 0.9940 / 3.53 | 0.9919 / 3.43 | 0.9877 / 2.40 |
| ef=100 | 0.9943 / 1.96 | 0.9940 / 3.64 | 0.9919 / 3.30 | 0.9877 / 2.42 |
| ef=128 | 0.9956 / 2.20 | 0.9955 / 4.10 | 0.9945 / 3.11 | 0.9920 / 2.72 |
| ef=200 | 0.9976 / 2.83 | 0.9975 / 4.47 | 0.9970 / 4.24 | 0.9960 / 3.80 |
| ef=300 | 0.9987 / 3.60 | 0.9986 / 5.72 | 0.9984 / 5.66 | 0.9979 / 4.86 |
| ef=512 | 0.9992 / 5.05 | 0.9992 / 7.71 | 0.9992 / 9.02 | 0.9990 / 6.46 |

**Unset resolves to `ef=100`.** Recall is identical to four decimals at every limit, and 100 is
this collection's `ef_construct`. The docs do not state the default; this is what it measures as.

### Binary quantization + rescore, at `ef=300` (matched to the baseline)

| Oversampling | limit 10 | limit 20 | limit 50 | limit 100 |
|---|---|---|---|---|
| 1x | 0.9277 / 0.80 | 0.9387 / 1.05 | 0.9453 / 1.19 | 0.9532 / 2.08 |
| 2x | 0.9786 / 0.81 | 0.9846 / 1.07 | 0.9857 / 1.22 | 0.9883 / 1.70 |
| 3x | 0.9904 / 0.83 | 0.9920 / 1.05 | 0.9939 / 1.22 | 0.9946 / 1.67 |
| 4x | 0.9940 / 0.85 | 0.9951 / 1.08 | 0.9961 / 1.28 | 0.9972 / 1.94 |
| **5x** | 0.9957 / 0.86 | 0.9966 / 1.48 | 0.9973 / 1.35 | **0.9982 / 1.96** |
| 6x | 0.9967 / 0.89 | 0.9975 / 1.30 | 0.9978 / 1.38 | 0.9988 / 2.23 |
| **8x** | 0.9975 / 0.95 | 0.9982 / 1.19 | **0.9987 / 1.58** | 0.9992 / 2.59 |
| **12x** | **0.9987 / 0.95** | **0.9986 / 1.45** | 0.9994 / 2.01 | 0.9997 / 3.10 |
| 16x | 0.9989 / 1.13 | 0.9988 / 1.81 | 0.9996 / 2.23 | 0.9997 / 4.41 |

Bold is the cheapest oversampling that matches or beats the `ef=300` unquantized baseline on
**both** recall and p95. Full sweep including 1.5x, the default-ef runs, and p50/p99 in
`results_k*.csv`.

## What the numbers say

**At matched `ef`, quantization wins on both axes at every limit — but it takes more
oversampling than 3x to get there.**

| limit | unquantized ef=300 | cheapest BQ that matches or beats it | speedup |
|---|---|---|---|
| 10 | 0.9987 / 3.60 ms | 12x, 0.9987 / 0.95 ms | ~3.8x |
| 20 | 0.9986 / 5.72 ms | 12x, 0.9986 / 1.45 ms | ~3.9x |
| 50 | 0.9984 / 5.66 ms | 8x, 0.9987 / 1.58 ms | ~3.6x |
| 100 | 0.9979 / 4.86 ms | 5x, 0.9982 / 1.96 ms | ~2.5x |

Speedups are given as approximate on purpose: recall reproduces to four decimals across runs, but the
larger latencies carry up to 1-3 ms of run-to-run spread (see `p95_spread_ms`), so the multiplier is
good to about half a turn, not to one decimal. The direction and rough size are solid; treat 3.8x as
"between three and four".

**Bigger `limit` needs less oversampling, not more.** At limit 100 the rescore pool is already
500 vectors at 5x, enough to fix the ranking; at limit 10 the same multiplier only buys 50, so
you need 12x to reach the same recall. Tune oversampling against your actual `limit`.

**The recommended 3x is a speed setting, not a parity setting.** It does not match a tuned
unquantized index at any limit here. What it does do is beat the *default-ef* baseline on both
axes at limit 100 (0.9946 / 1.70 ms against 0.9877 / 2.40 ms). If you want parity with a tuned
index, go to 5x-12x depending on limit.

**Oversampling is much cheaper than `ef` for the same recall.** Reading the two frontiers: to
buy 0.9987 recall at limit 10, `ef` costs 3.60 ms and oversampling costs 0.95 ms. Both knobs walk
more candidates; only one of them walks them as 1-bit vectors. Note that this is a latency win,
not a memory win: rescoring keeps the original float vectors resident, so this configuration
stores the 1-bit copy *in addition to* them.

**Rescore is not optional, and oversampling is not a substitute.** With rescoring off, binary
quantization returns 0.67-0.69 recall, and sweeping oversampling from 1x to 16x moves it by
**+0.0003 to +0.0013** - a flat line. Turning rescoring on at 1x, before any extra candidates are
fetched, is worth **+0.251 to +0.262** recall, and costs between -0.10 and +0.13 ms of p95, which is
inside the noise. Oversampling widens the candidate pool; rescoring is what reads the original
vectors, and only one of those two recovers accuracy.

| limit | rescore off, 1x | rescore on, 1x | recall gained | p95 delta | rescore off, 1x -> 16x |
|---|---|---|---|---|---|
| 10 | 0.6725 / 0.69 ms | 0.9238 / 0.70 ms | +0.2513 | +0.01 ms | +0.0011 |
| 20 | 0.6820 / 0.89 ms | 0.9354 / 0.80 ms | +0.2534 | -0.09 ms | +0.0013 |
| 50 | 0.6830 / 1.01 ms | 0.9419 / 0.90 ms | +0.2589 | -0.10 ms | +0.0012 |
| 100 | 0.6873 / 1.07 ms | 0.9496 / 1.19 ms | +0.2623 | +0.13 ms | +0.0003 |

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
- **Timing.** Each configuration runs three passes of 1,000 sequential single queries over local gRPC,
  after a 200-query warmup, client-side wall clock, nothing else on the host. Every reported percentile
  is the median across the three passes, and `p95_spread_ms` records max minus min so the variance stays
  visible. One pass is not enough: a single contended window moved a p95 from 3.76 to 8.61 ms and shifted
  a published speedup from 2.7x to 3.9x. Recall comes from one pass because it is deterministic - it
  reproduces to four decimals across every run.

## What these numbers do not cover

- **Latency resolution is coarser than the digits suggest.** Cheap configurations are steady
  (`p95_spread_ms` of 0.03-0.15), but the expensive ones are not: ef=512 at limit 50 spread 2.75 ms,
  16x at limit 100 spread 3.25 ms. Read differences under ~0.5 ms as noise, and the frontier gaps
  (1-4 ms) as real. `p50_ms` is steadier if you need finer resolution.
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

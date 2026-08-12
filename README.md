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
limit, limits 10/20/50/100), adds latency percentiles, and adds an `hnsw_ef` sweep on **both**
sides so the comparison has a control that holds.

**[Read the report](https://claude.ai/code/artifact/9cad4724-5673-415f-8b2b-95df43633db1)**
(charts, full tables, method) or open `report.html` locally.

## Setup

100,000 DBpedia entity vectors, OpenAI `text-embedding-ada-002` (1536d, cosine), 1,000 held-out
queries, binary quantization with vectors in RAM, Qdrant 1.19.0 on an Apple M5. Ground truth is
exact KNN on the same collection. Every cell is `recall / p95 ms`, and each latency percentile is
the median of three passes.

## The mechanism the docs leave out

`oversampling` and `hnsw_ef` both control how many candidates a search considers, and no Qdrant
page mentions them together. The quantization docs define oversampling without referencing
`hnsw_ef`; the [search-parameters docs](https://qdrant.tech/documentation/ops-optimization/optimize/#fine-tuning-search-parameters)
define `hnsw_ef` as "number of neighbors to visit during search" without referencing oversampling.
The only link between them is a negative rule in the `qdrant-search-quality` agent skill: *"Set
`hnsw_ef` lower than results requested (guaranteed bad recall)."* Since oversampling raises the
number of results requested from the index to `limit x oversampling`, that rule implies `hnsw_ef`
has to track oversampling.

Measured, the engine already reconciles them:

```
graph walk width = max(hnsw_ef, limit x oversampling)
rescore set size = limit x oversampling
```

At limit 10 with the default `ef` of 100, that predicts oversampling below 10x changes nothing
about the graph walk and only enlarges the rescore set. It does:

| oversampling | pool | walk, `ef` unset | recall | walk, `ef=pool` | recall |
|---|---|---|---|---|---|
| 1x | 10 | 100 | 0.9238 | 10 | 0.8339 |
| 2x | 20 | 100 | 0.9747 | 20 | 0.9321 |
| 4x | 40 | 100 | 0.9900 | 40 | 0.9755 |
| 8x | 80 | 100 | 0.9931 | 80 | 0.9905 |
| **12x** | **120** | **120** | **0.9947** | **120** | **0.9947** |
| **16x** | **160** | **160** | **0.9966** | **160** | **0.9966** |

Where the two configurations imply the same walk width, recall is identical to four decimals:
**25 of 25** such rows across all four limits. Where `ef=pool` falls below the default and
therefore narrows the walk, it is worse: **15 of 15**. No exceptions.

Two practical consequences:

- **You cannot starve the pool by leaving `hnsw_ef` alone.** The pool wins the `max`, so the
  skill's warning does not bite on the oversampling path. You *can* starve it by explicitly
  setting `hnsw_ef` below the default, which is what the `ef=pool` column does at small pools.
- **At small limits, oversampling does something other than you would guess.** From 1x to 8x at
  limit 10, recall climbs 0.9238 to 0.9931 while the graph walk never moves off 100. The gain comes
  entirely from rescoring more of the candidates already found, which is also why latency there is
  nearly flat.

## Comparing quantized against unquantized

Oversampling pre-selects `limit x oversampling` candidates, so the matched unquantized run is one
that walks the same size pool: `ef = limit x oversampling`. Then the only difference left is 1-bit
versus float distance computation. Compared that way, **unquantized wins on recall at every pool
size** - which is the ordinary compression tradeoff, exactly as the docs describe it.

**limit 100, same pool on both sides:**

| pool | BQ recall / p95 | unquantized recall / p95 | recall gap |
|---|---|---|---|
| 100 | 0.9496 / 1.17 ms | 0.9877 / 2.60 ms | +0.0381 |
| 200 | 0.9871 / 1.39 ms | 0.9960 / 4.55 ms | +0.0089 |
| 400 | 0.9972 / 1.74 ms | 0.9987 / 6.13 ms | +0.0015 |
| 800 | 0.9992 / 2.39 ms | 0.9994 / 8.00 ms | +0.0002 |
| 1600 | 0.9997 / 3.67 ms | 0.9997 / 11.23 ms | +0.0000 |

The gap closes as the pool grows, because a bigger rescore set recovers more of what compression
lost. By 1600 candidates the two are identical on recall and quantization is 3.1x faster.

So the useful question is not "which is better at the same pool" - it is **what does each cost to
reach the recall you need.**

| limit | recall target | binary quantization | unquantized | speedup |
|---|---|---|---|---|
| 10 | 0.99 | 8x, 0.9905 / 0.69 ms | ef=80, 0.9920 / 1.67 ms | 2.4x |
| 10 | 0.995 | 16x, 0.9966 / 0.86 ms | ef=120, 0.9953 / 2.04 ms | 2.4x |
| 50 | 0.99 | 3x, 0.9913 / 1.10 ms | ef=100, 0.9919 / 2.18 ms | 2.0x |
| 50 | 0.999 | 12x, 0.9994 / 2.01 ms | ef=400, 0.9990 / 5.39 ms | 2.7x |
| 100 | 0.99 | 3x, 0.9946 / 1.56 ms | ef=150, 0.9938 / 3.47 ms | 2.2x |
| 100 | 0.995 | 4x, 0.9972 / 1.74 ms | ef=200, 0.9960 / 4.55 ms | 2.6x |
| 100 | 0.999 | 8x, 0.9992 / 2.39 ms | ef=500, 0.9990 / 6.59 ms | 2.8x |

**Binary quantization with rescoring reaches any recall target on this dataset 2.0x to 2.8x faster
than an unquantized index tuned to the same target.** It gets there by affording a much larger
candidate pool for the same latency budget, not by being more accurate per candidate.

This is a latency win, not a memory win. Rescoring reads the original float vectors, so they stay
resident and the 1-bit copy is stored *in addition to* them.

## What else the numbers say

**Rescore is not optional, and oversampling is not a substitute.** With rescoring off, binary
quantization returns 0.67-0.69 recall, and sweeping oversampling from 1x to 16x moves it by
**+0.0003 to +0.0013** - a flat line. Turning rescoring on at 1x, before any extra candidates are
fetched, is worth **+0.251 to +0.262** recall and costs between -0.12 and +0.04 ms of p95, which is
inside the noise.

| limit | rescore off, 1x | rescore on, 1x | recall gained | p95 delta | rescore off, 1x -> 16x |
|---|---|---|---|---|---|
| 10 | 0.6725 / 0.65 ms | 0.9238 / 0.64 ms | +0.2513 | -0.01 ms | +0.0011 |
| 20 | 0.6820 / 0.82 ms | 0.9354 / 0.70 ms | +0.2534 | -0.12 ms | +0.0013 |
| 50 | 0.6830 / 0.96 ms | 0.9419 / 0.93 ms | +0.2589 | -0.03 ms | +0.0012 |
| 100 | 0.6873 / 1.11 ms | 0.9496 / 1.15 ms | +0.2623 | +0.04 ms | +0.0003 |

**The default `hnsw_ef` is 100.** Leaving it unset gives recall identical to four decimals to an
explicit `ef=100` at every limit, which is this collection's `ef_construct`. The docs do not state
the default; this is what it measures as.

**Bigger `limit` needs less oversampling.** Reaching 0.99 takes 8x at limit 10 but 3x at limit 100,
because the multiplier applies to a larger base and the rescore pass already has more to work with.
Tune oversampling against your actual `limit`, not as a fixed number.

**3x is a reasonable default.** It clears 0.99 recall at limits 50 and 100, and at limit 100 it is
2.2x faster than an unquantized index tuned to the same recall. At limit 10 it reaches 0.9862, and
you would want more than 3x there if you need three nines.

## Run it

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt

# Qdrant 1.19.0 - any local instance works; Docker or the release binary
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:v1.19.0

./.venv/bin/python bench.py ingest                    # downloads ~1 GB, indexes 100k, waits for green
for k in 10 20 50 100; do ./.venv/bin/python bench.py bench $k 3; done
./.venv/bin/python summarize.py                       # the tables above
```

`bench.py bench <limit> <reps>` runs 57 configurations per limit. Add one by appending to `CONFIGS`;
scalar quantization is a one-line change there. `python test_bench.py` checks the recall math, the
CSV quoting, and that the published files still agree with each other.

## Method

- **Ground truth.** Exact KNN over the same collection (`exact=true`, quantization ignored),
  following Qdrant's
  [ANN-recall method](https://qdrant.tech/documentation/tutorials-search-engineering/ann-recall/#automate-in-ci-with-python).
  Recall is the overlap of the approximate and exact id sets divided by the limit, averaged over
  1,000 queries. Every target is reachable by construction, so this isolates index and quantization
  loss. It says nothing about whether the embedding model retrieves the right thing.
- **Index.** m=16, ef_construct=100, binary quantization with `always_ram`, original vectors in RAM.
  Green and fully indexed before any query is timed.
- **Controls.** Three of them, because the first one was wrong. A fixed `ef` is not a control: below
  `limit x oversampling` it is ignored, and above it it hands the quantized side a wider walk than
  the unquantized side gets. The pool-matched runs (`ef=pool`) are the control the conclusions rest
  on; the fixed `ef=300` and default-`ef` runs are kept because they are what exposes the mechanism.
- **Timing.** Three passes of 1,000 sequential single queries per configuration over local gRPC,
  after a 200-query warmup, client-side wall clock, nothing else on the host. Every percentile is
  the median across passes; `p95_spread_ms` records max minus min. One pass is not enough: a single
  contended window moved a p95 from 3.76 to 8.61 ms. Recall comes from one pass because it is
  deterministic - it reproduces to four decimals across every run.

## What these numbers do not cover

- **Latency resolution is coarser than the digits suggest.** Cheap configurations are steady
  (`p95_spread_ms` of 0.03-0.15), the expensive ones are not (up to 3.25 ms). Read differences under
  ~0.5 ms as noise and the frontier gaps of 1-8 ms as real. `p50_ms` is steadier.
- **Transport is included.** Roughly 0.3 ms of every figure is client and gRPC overhead.
- **Rescoring from disk.** Originals are in RAM here. The docs warn that disk-backed rescoring is
  much slower, and this run does not measure it.
- **Concurrency.** Sequential queries only. Throughput under load is a separate measurement.
- **One model, one dimensionality.** Binary quantization is sensitive to how a model distributes its
  components, and 1536d OpenAI vectors are close to its best case. The OpenAI article sweeps
  512-3072 dimensions across two models; this repo does not. Re-run before quoting these numbers for
  a different model.
- **One node, 100k vectors.** Both curves shift with collection size and shard count.

## License

Apache-2.0. See [LICENSE](LICENSE).

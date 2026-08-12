"""Merge the per-limit result files and print the comparisons the README quotes.

usage: python summarize.py
"""
import json, glob, re

LIMITS = [10, 20, 50, 100]
rows = []
for f in sorted(glob.glob("results_k*.json"), key=lambda p: int(re.search(r"k(\d+)", p).group(1))):
    rows += json.load(open(f))

by = lambda **kw: [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
get = lambda **kw: (by(**kw) or [None])[0]
cell = lambda r: f"{r['recall']:.4f} / {r['p95_ms']:.2f}" if r else "-"


def table(title, picker):
    print(f"\n## {title}\n")
    print("| Configuration | " + " | ".join(f"limit {l}" for l in LIMITS) + " |")
    print("|---" * (len(LIMITS) + 1) + "|")
    for label, kw in picker:
        print(f"| {label} | " + " | ".join(cell(get(limit=l, **kw)) for l in LIMITS) + " |")


print("recall / p95 ms, 1,000 queries per cell")

table("Unquantized HNSW: how much of the gap is just ef?", [
    ("no quantization, ef unset", dict(exact=False, hnsw_ef=None, oversampling=None)),
] + [(f"no quantization, ef={ef}", dict(hnsw_ef=ef, oversampling=None)) for ef in (100, 128, 200, 300, 512)])

table("Binary quantization at the engine default ef", [
    ("rescore off", dict(rescore=False)),
] + [(f"rescore, {o}x", dict(rescore=True, oversampling=float(o), hnsw_ef=None))
     for o in (1, 1.5, 2, 3, 4, 5, 6, 8, 12, 16)])

table("Binary quantization at ef=300 (matched to the baseline)", [
    (f"rescore, {o}x, ef=300", dict(rescore=True, oversampling=float(o), hnsw_ef=300))
    for o in (1, 1.5, 2, 3, 4, 5, 6, 8, 12, 16)
])

print("\n## Matched-ef verdict (ef=300 both sides)\n")
for l in LIMITS:
    base = get(limit=l, hnsw_ef=300, oversampling=None)
    beats = [r for r in by(limit=l, hnsw_ef=300, rescore=True)
             if r["recall"] >= base["recall"] and r["p95_ms"] <= base["p95_ms"]]
    cheapest = min(beats, key=lambda r: r["oversampling"]) if beats else None
    print(f"limit {l:>3}: baseline {base['recall']:.4f} / {base['p95_ms']:.2f} ms  ->  "
          + (f"BQ {cheapest['oversampling']:g}x wins on both ({cheapest['recall']:.4f} / {cheapest['p95_ms']:.2f} ms)"
             if cheapest else "no BQ setting wins on both axes"))

"""Guards the two things that would silently corrupt published numbers:
the recall math and the CSV quoting. Run: python test_bench.py
"""
import csv, json, os, tempfile

import bench


def test_recall():
    truth = [{1, 2, 3, 4}, {5, 6, 7, 8}]
    assert bench.recall_at_k(truth, truth, 4) == 1.0
    assert bench.recall_at_k([{1, 2, 9, 10}, {5, 6, 7, 8}], truth, 4) == 0.75
    assert bench.recall_at_k([set(), set()], truth, 4) == 0.0


def test_csv_quotes_commas():
    rows = [dict(config="HNSW, no quantization", oversampling=None, recall=0.9943,
                 p50_ms=1.652, p95_ms=1.87, p99_ms=2.024, mean_ms=1.635)]
    path = os.path.join(tempfile.mkdtemp(), "r.csv")
    bench.write_csv(rows, path)
    got = list(csv.DictReader(open(path)))
    assert len(got) == 1 and set(got[0]) == set(bench.FIELDS)
    assert got[0]["config"] == "HNSW, no quantization"
    assert float(got[0]["p95_ms"]) == 1.87


def test_published_results_match_json():
    for k in (10, 20, 50, 100):
        rows = list(csv.DictReader(open(f"results_k{k}.csv")))
        ref = json.load(open(f"results_k{k}.json"))
        assert len(rows) == len(ref) == len(bench.CONFIGS)
        for a, b in zip(rows, ref):
            assert a["config"] == b["config"]
            assert float(a["recall"]) == b["recall"]
            assert float(a["p95_ms"]) == b["p95_ms"]
            assert int(a["limit"]) == b["limit"] == k


def test_ground_truth_is_first():
    """Recall is scored against row 0, so the exact run must lead every file."""
    assert bench.CONFIGS[0][1]["exact"] is True
    for k in (10, 20, 50, 100):
        first = json.load(open(f"results_k{k}.json"))[0]
        assert first["exact"] is True and first["recall"] == 1.0


if __name__ == "__main__":
    test_recall()
    test_csv_quotes_commas()
    test_published_results_match_json()
    test_ground_truth_is_first()
    print("ok")

"""Run the bundled labeling dataset through verqen-labeler and record metrics.

Usage::

    python benchmarks/run_benchmark.py --checkpoint "$VERQEN_CHECKPOINT"
    python benchmarks/run_benchmark.py --limit 20 --fail-under 0.9

Loads ``labeling_dataset.jsonl`` (60 rows across 5 topics), labels every
text with a live labeler, and writes a JSON report to ``benchmarks/results/``
with:

* accuracy, auto-label rate, human review rate
* empirical coverage of conformal sets (expected label inside the set)
* average conformal set size
* throughput (labels/sec) and latency p50 / p95

The bundled dataset is a starter regression suite. The official VerQen
benchmark numbers in the README come from the full evaluation suite.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "labeling_dataset.jsonl"
RESULTS_DIR = HERE / "results"

sys.path.insert(0, str(HERE.parent))  # allow running without pip install

from verqen_labeler import Labeler  # noqa: E402

CATEGORIES = ["sports", "politics", "technology", "entertainment", "finance"]


def load_dataset(path: Path = DEFAULT_DATASET) -> list[dict]:
    """Load and validate the JSONL dataset (fields: id, text, expected_label)."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = {"id", "text", "expected_label"} - row.keys()
            if missing:
                raise ValueError(f"{path.name}:{lineno}: missing fields {sorted(missing)}")
            if row["expected_label"] not in CATEGORIES:
                raise ValueError(
                    f"{path.name}:{lineno}: unknown expected_label {row['expected_label']!r}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: dataset is empty")
    return rows


def check_row(labeler: Labeler, row: dict) -> dict:
    """Run one dataset row through the labeler and time it."""
    start = time.perf_counter()
    result = labeler.label(row["text"])
    elapsed_ms = (time.perf_counter() - start) * 1000
    conformal_set = list(result.conformal_set)
    return {
        "id": row["id"],
        "expected_label": row["expected_label"],
        "predicted": result.label,
        "correct": result.label == row["expected_label"],
        "auto_labeled": not result.needs_human_review,
        "covered": row["expected_label"] in conformal_set,
        "set_size": len(conformal_set),
        "confidence": round(result.confidence, 4),
        "latency_ms": round(elapsed_ms, 3),
    }


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), -(-int(p * len(ordered)) // 100)))
    return ordered[rank - 1]


def summarize(records: list[dict]) -> dict:
    """Aggregate per-record results into a metrics report (no wall time)."""
    total = len(records)
    if total == 0:
        raise ValueError("no records to summarize")
    correct = sum(1 for record in records if record["correct"])
    auto = sum(1 for record in records if record["auto_labeled"])
    covered = sum(1 for record in records if record["covered"])
    latencies = [record["latency_ms"] for record in records]
    return {
        "total": total,
        "accuracy": round(correct / total, 4),
        "auto_label_rate": round(auto / total, 4),
        "human_review_rate": round((total - auto) / total, 4),
        "empirical_coverage": round(covered / total, 4),
        "avg_set_size": round(sum(record["set_size"] for record in records) / total, 4),
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 3),
            "p95": round(percentile(latencies, 95), 3),
            "max": round(max(latencies), 3),
        },
        "errors": [record["id"] for record in records if not record["correct"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None, help="VerQen checkpoint path")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=0, help="only run the first N rows")
    parser.add_argument("--fail-under", type=float, default=None,
                        help="exit 3 when accuracy is below this value")
    args = parser.parse_args(argv)

    rows = load_dataset(args.dataset)
    if args.limit > 0:
        rows = rows[: args.limit]

    try:
        labeler = Labeler(categories=CATEGORIES, checkpoint=args.checkpoint)
    except (ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    started = time.time()
    records = []
    for index, row in enumerate(rows, start=1):
        records.append(check_row(labeler, row))
        if index % 20 == 0 or index == len(rows):
            print(f"  {index}/{len(rows)}", flush=True)
    wall_seconds = max(0.001, time.time() - started)

    metrics = summarize(records)
    metrics["throughput_per_second"] = round(len(records) / wall_seconds, 2)
    report = {
        "dataset": str(args.dataset.name),
        "rows": len(records),
        "wall_seconds": round(wall_seconds, 2),
        "metrics": metrics,
        "records": records,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"benchmark-{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    print(f"dataset      {report['dataset']} ({report['rows']} rows)")
    print(f"accuracy     {metrics['accuracy']:.1%}")
    print(f"auto label   {metrics['auto_label_rate']:.1%}")
    print(f"human review {metrics['human_review_rate']:.1%}")
    print(f"coverage     {metrics['empirical_coverage']:.1%}  avg set {metrics['avg_set_size']}")
    print(f"throughput   {metrics['throughput_per_second']} labels/sec")
    print(f"latency p50  {metrics['latency_ms']['p50']} ms  p95 {metrics['latency_ms']['p95']} ms")
    print(f"report       {out_path}")
    if metrics["errors"]:
        print(f"misses ({len(metrics['errors'])}): {', '.join(metrics['errors'][:20])}")

    if args.fail_under is not None and metrics["accuracy"] < args.fail_under:
        print(
            f"FAIL: accuracy {metrics['accuracy']:.3f} < {args.fail_under:.3f}",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

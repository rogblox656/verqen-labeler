"""Quickstart: label a few texts with certified conformal sets.

Requires a local checkpoint::

    export VERQEN_CHECKPOINT=/path/to/verqen-checkpoint
    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow running without pip install

from verqen_labeler import Labeler  # noqa: E402

CATEGORIES = ["sports", "politics", "technology", "entertainment", "finance"]

TEXTS = [
    "The striker scored twice in the second half to seal the win.",
    "Lawmakers debated the new immigration bill late into the evening.",
    "The central bank raised interest rates again this quarter.",
]


def main() -> None:
    labeler = Labeler(categories=CATEGORIES)  # reads VERQEN_CHECKPOINT

    for text in TEXTS:
        result = labeler.label(text)
        flag = "review" if result.needs_human_review else "auto"
        print(f"[{flag}] {result.label:15} {result.confidence:.2f} set={result.conformal_set}")
        print(f"       {text}")

    results, report = labeler.label_dataset(TEXTS)
    print()
    print(f"total={report.total}  auto={report.auto_labeled}  review={report.human_review_needed}")
    print(
        f"estimated_accuracy={report.estimated_accuracy:.3f}  "
        f"coverage={report.coverage:.2f}  empirical_coverage={report.empirical_coverage:.2f}"
    )
    print(
        f"throughput={report.throughput_per_second:.1f} labels/sec  "
        f"cost=${report.cost_usd:.6f}"
    )


if __name__ == "__main__":
    main()

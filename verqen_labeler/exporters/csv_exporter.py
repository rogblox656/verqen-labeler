"""CSV exporter: one row per label, stdlib ``csv`` module only."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["FIELDS", "export_csv"]

FIELDS = ["text", "label", "confidence", "conformal_set", "needs_human_review", "latency_ms"]


def export_csv(results: Sequence[Any], path: str | Path) -> Path:
    """Write *results* as CSV with a header row.

    ``conformal_set`` is stored as a JSON array string (round-trip safe)
    and ``needs_human_review`` as ``true`` / ``false``.
    """
    target = Path(path)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "text": result.text,
                    "label": result.label,
                    "confidence": f"{result.confidence:.6f}",
                    "conformal_set": json.dumps(list(result.conformal_set)),
                    "needs_human_review": "true" if result.needs_human_review else "false",
                    "latency_ms": f"{result.latency_ms:.3f}",
                }
            )
    return target

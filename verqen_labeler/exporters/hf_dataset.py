"""Hugging Face datasets exporter: plain JSONL, no ``datasets`` dependency.

The file written here is the JSON Lines format ``datasets`` loads directly.
After exporting::

    from datasets import load_dataset

    dataset = load_dataset("json", data_files="labels.jsonl")
    dataset.push_to_hub("your-org/your-dataset")

``datasets`` is only needed on the publishing side (install with
``pip install "verqen-labeler[datasets]"``); writing the file needs nothing
but the standard library.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["export_hf"]


def export_hf(results: Sequence[Any], path: str | Path) -> Path:
    """Write *results* as JSONL: one label object per line."""
    target = Path(path)
    with target.open("w", encoding="utf-8") as handle:
        for result in results:
            record = {
                "text": result.text,
                "label": result.label,
                "confidence": result.confidence,
                "conformal_set": list(result.conformal_set),
                "needs_human_review": result.needs_human_review,
                "latency_ms": result.latency_ms,
            }
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return target

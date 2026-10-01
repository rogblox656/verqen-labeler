"""COCO-style exporter for text-as-category labeling (pure JSON, no pycocotools).

The schema mirrors COCO's ``images`` / ``annotations`` / ``categories``
blocks. Each text sample becomes an ``image`` entry (the raw text rides
inline under ``text`` with a stable ``file_name``), its predicted label
becomes one annotation with a degenerate ``bbox`` (text samples have no
pixel box), and ``categories`` lists every label that appears. Standard
COCO tooling can index the file; this writer needs only ``json``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["export_coco"]


def export_coco(results: Sequence[Any], path: str | Path) -> Path:
    """Write *results* as a COCO-style JSON annotation file."""
    target = Path(path)

    category_ids: dict[str, int] = {}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    for index, result in enumerate(results):
        if result.label not in category_ids:
            category_ids[result.label] = len(category_ids) + 1
        image_id = index + 1
        images.append(
            {
                "id": image_id,
                "file_name": f"text-{image_id:04d}.txt",
                "text": result.text,
            }
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": category_ids[result.label],
                "bbox": [0, 0, 0, 0],
                "area": 0,
                "iscrowd": 0,
                "confidence": round(result.confidence, 6),
            }
        )

    payload = {
        "info": {"description": "verqen-labeler export (text as category)", "version": "1.0"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name}
            for name, category_id in sorted(category_ids.items(), key=lambda pair: pair[1])
        ],
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    return target

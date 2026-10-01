"""Export LabelResults to common formats.

Registry-based: ``export(results, path, fmt)``. Every writer is pure Python
with no optional dependencies: ``csv`` uses the stdlib, ``hf`` writes JSONL
that ``datasets`` can load later, and ``coco`` / ``label_studio`` write
plain JSON.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

from verqen_labeler.exporters import coco, csv_exporter, hf_dataset, label_studio

__all__ = ["available_formats", "export"]

_WRITERS: dict[str, Callable[..., Path]] = {
    "csv": csv_exporter.export_csv,
    "coco": coco.export_coco,
    "hf": hf_dataset.export_hf,
    "label_studio": label_studio.export_label_studio,
}


def available_formats() -> list[str]:
    """Sorted registry keys accepted by :func:`export`."""
    return sorted(_WRITERS)


def export(results: Sequence[Any], path: str | Path, fmt: str) -> Path:
    """Write *results* (LabelResult objects) to *path* in format *fmt*.

    Formats: ``csv``, ``coco``, ``hf``, ``label_studio``. Parent directories
    are created as needed and the ``Path`` is returned. Raises
    ``ValueError`` for an unknown format.
    """
    writer = _WRITERS.get(fmt)
    if writer is None:
        raise ValueError(
            f"unknown export format {fmt!r}; available: {', '.join(available_formats())}"
        )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    writer(results, target)
    return target

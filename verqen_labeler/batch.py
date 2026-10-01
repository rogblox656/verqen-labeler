"""High-throughput labeling: batch evaluation with wall-clock timing."""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verqen_labeler.labeler import Labeler

__all__ = ["evaluate_many"]


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100); 0.0 for an empty list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), -(-int(p * len(ordered)) // 100)))
    return ordered[rank - 1]


def evaluate_many(
    labeler: "Labeler",
    texts: Sequence[str],
    *,
    context: str = "",
    num_workers: int = 1,
) -> tuple[list, dict]:
    """Label *texts* in input order and measure wall-clock throughput.

    ``num_workers=1`` labels sequentially. ``num_workers>1`` runs
    ``labeler.label`` in a thread pool; the labeler serializes engine access
    with its own lock, so the engine is never called concurrently.

    Returns ``(results, timing)`` where ``timing`` holds ``count``,
    ``total_seconds``, ``throughput_per_second`` and ``latency_ms``
    (p50/p95/max of the per-sample latencies, in milliseconds).
    """
    if num_workers < 1:
        raise ValueError("num_workers must be >= 1")
    items = list(texts)

    start = time.perf_counter()
    if num_workers == 1:
        results = [labeler.label(text, context) for text in items]
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            results = list(pool.map(lambda text: labeler.label(text, context), items))
    elapsed = time.perf_counter() - start

    latencies = [result.latency_ms for result in results]
    timing = {
        "count": len(results),
        "total_seconds": elapsed,
        "throughput_per_second": len(results) / max(0.001, elapsed),
        "latency_ms": {
            "p50": _percentile(latencies, 50),
            "p95": _percentile(latencies, 95),
            "max": max(latencies) if latencies else 0.0,
        },
    }
    return results, timing

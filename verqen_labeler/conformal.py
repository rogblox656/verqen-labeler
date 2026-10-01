"""Conformal prediction helpers for certified label sets.

A VerQen decision returns a ``prediction_set``: the set of labels certified
to contain the true label at the target coverage (default 95%). These
helpers turn engine decisions into label sets, account for realized
coverage, and decide when a sample needs a human.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "extract_label_set",
    "needs_review",
    "set_size_stats",
    "covered_fraction",
    "empirical_coverage",
]


def extract_label_set(decision: Mapping[str, Any]) -> list[str]:
    """Certified label set from one engine decision.

    The engine reports ``prediction_set`` (present when the checkpoint was
    calibrated). When the key is missing or empty, fall back to the single
    predicted label.
    """
    selected = str(decision.get("selected", ""))
    return [str(label) for label in (decision.get("prediction_set") or [selected])]


def needs_review(conformal_set: Sequence[str], confidence: float, threshold: float) -> bool:
    """True when a sample should go to a human: the set holds more than one
    label, or the confidence is below *threshold*.
    """
    return len(conformal_set) > 1 or confidence < threshold


def set_size_stats(sets: Sequence[Sequence[str]]) -> dict[str, float]:
    """Aggregate label-set sizes: ``count``, ``avg``, ``min``, ``max`` and
    ``single_rate`` (fraction of sets with exactly one label).

    Empty input gives all zeros.
    """
    if not sets:
        return {"count": 0, "avg": 0.0, "min": 0, "max": 0, "single_rate": 0.0}
    sizes = [len(item) for item in sets]
    return {
        "count": len(sizes),
        "avg": sum(sizes) / len(sizes),
        "min": min(sizes),
        "max": max(sizes),
        "single_rate": sum(1 for size in sizes if size == 1) / len(sizes),
    }


def covered_fraction(conformal_sets: Sequence[Sequence[str]], ground_truth: Sequence[str]) -> float:
    """Fraction of *ground_truth* labels contained in the paired conformal sets.

    Raises ``ValueError`` on a length mismatch; empty input gives 0.0.
    """
    if len(conformal_sets) != len(ground_truth):
        raise ValueError(
            f"length mismatch: {len(conformal_sets)} sets vs {len(ground_truth)} labels"
        )
    if not conformal_sets:
        return 0.0
    hits = sum(1 for item, truth in zip(conformal_sets, ground_truth) if truth in item)
    return hits / len(conformal_sets)


def empirical_coverage(results: Sequence[Any], ground_truth: Sequence[str] | None = None) -> float:
    """Fraction of labels covered by their conformal set over *results*.

    With *ground_truth* (one known label per result), this is the classic
    conformal coverage: how often the true label lands inside the certified
    set. Without it, each result's own predicted label is the reference,
    which measures set integrity (1.0 when every prediction sits in its set).
    """
    sets = [list(result.conformal_set) for result in results]
    if ground_truth is not None:
        reference = list(ground_truth)
    else:
        reference = [result.label for result in results]
    return covered_fraction(sets, reference)

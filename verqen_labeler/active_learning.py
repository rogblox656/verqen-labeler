"""Informative-sample selection for active labeling loops.

Pure functions over engine decision dicts (the output of ``engine.decide``):
no engine, no I/O. Rank an unlabeled pool by uncertainty and label the most
informative samples first.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["entropy", "normalized_entropy", "uncertainty_score", "rank_samples", "pick_top_k"]


def entropy(probabilities: Mapping[str, float]) -> float:
    """Shannon entropy in bits (log2) of a probability distribution.

    Zero-probability entries contribute nothing; empty input gives 0.0.
    """
    total = 0.0
    for raw in probabilities.values():
        value = float(raw)
        if value > 0.0:
            total += value * math.log2(value)
    return -total


def normalized_entropy(probabilities: Mapping[str, float]) -> float:
    """Entropy scaled to [0, 1] by ``log2(number of options)``.

    0.0 when there are fewer than 2 options.
    """
    count = len(probabilities)
    if count <= 1:
        return 0.0
    return entropy(probabilities) / math.log2(count)


def uncertainty_score(decision: Mapping[str, Any]) -> float:
    """How informative a sample is, 0.0 (certain) to 1.0 (maximally uncertain).

    Mean of three terms:
    1. normalized entropy of ``probabilities``,
    2. ``1 - margin`` with the margin clamped to [0, 1],
    3. label-set size: ``(len(prediction_set) - 1) / 3`` clamped to [0, 1].

    A decision without ``probabilities`` contributes 0 for term 1; a
    decision without ``prediction_set`` falls back to ``[selected]``.
    """
    probabilities = decision.get("probabilities") or {}
    raw_margin = decision.get("margin")
    margin = float(raw_margin) if raw_margin is not None else 0.0
    label_set = decision.get("prediction_set")
    if not label_set:
        selected = decision.get("selected")
        label_set = [selected] if selected else []

    term_entropy = normalized_entropy(probabilities)
    term_margin = 1.0 - min(1.0, max(0.0, margin))
    term_set = min(1.0, max(0.0, (len(label_set) - 1) / 3.0))
    return (term_entropy + term_margin + term_set) / 3.0


def rank_samples(samples: Sequence[Mapping[str, Any]]) -> list[int]:
    """Indices of *samples* ordered by descending uncertainty.

    Ties keep the lower original index first.
    """
    return sorted(
        range(len(samples)),
        key=lambda index: (-uncertainty_score(samples[index]), index),
    )


def pick_top_k(samples: Sequence[Mapping[str, Any]], k: int) -> list[int]:
    """Indices of the *k* most uncertain samples (``k`` clamped to the pool size)."""
    if k < 0:
        raise ValueError("k must be >= 0")
    return rank_samples(samples)[:k]

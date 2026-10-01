"""Pure label-quality metrics: accuracy, calibration (ECE), coverage, agreement.

All functions are pure: plain sequences in, numbers out, no engine and no
I/O. ``coverage`` is :func:`verqen_labeler.conformal.covered_fraction`
re-exported for a single import point.
"""

from __future__ import annotations

from collections.abc import Sequence

from verqen_labeler.conformal import covered_fraction as coverage

__all__ = [
    "accuracy",
    "expected_calibration_error",
    "calibration_bins",
    "coverage",
    "agreement_rate",
]


def accuracy(predictions: Sequence[str], ground_truth: Sequence[str]) -> float:
    """Fraction of predictions equal to their ground-truth label.

    Meant for the labeled subset of a batch. Raises ``ValueError`` on a
    length mismatch or empty input.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"length mismatch: {len(predictions)} predictions vs {len(ground_truth)} labels"
        )
    if not predictions:
        raise ValueError("accuracy needs at least one sample")
    hits = sum(1 for predicted, truth in zip(predictions, ground_truth) if predicted == truth)
    return hits / len(predictions)


def calibration_bins(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Equal-width confidence bins over [0, 1]; only non-empty bins returned.

    Each bin is ``{"bin", "lower", "upper", "count", "avg_confidence",
    "accuracy"}``. The top bin includes confidence 1.0. Raises
    ``ValueError`` on bad input (length mismatch, empty, out-of-range
    confidence, ``n_bins < 1``).
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    if len(confidences) != len(correct):
        raise ValueError(
            f"length mismatch: {len(confidences)} confidences vs {len(correct)} outcomes"
        )
    if not confidences:
        raise ValueError("calibration needs at least one sample")

    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for raw_confidence, raw_correct in zip(confidences, correct):
        value = float(raw_confidence)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence {value} outside [0, 1]")
        index = min(n_bins - 1, int(value * n_bins))
        buckets[index].append((value, bool(raw_correct)))

    bins: list[dict[str, float]] = []
    for index, items in enumerate(buckets):
        if not items:
            continue
        count = len(items)
        bins.append(
            {
                "bin": index,
                "lower": index / n_bins,
                "upper": (index + 1) / n_bins,
                "count": count,
                "avg_confidence": sum(value for value, _ in items) / count,
                "accuracy": sum(1 for _, ok in items if ok) / count,
            }
        )
    return bins


def expected_calibration_error(
    confidences: Sequence[float],
    correct: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Expected calibration error: sum over bins of

    ``(bin count / n) * |bin accuracy - bin confidence|``.
    """
    bins = calibration_bins(confidences, correct, n_bins)
    total = len(confidences)
    return sum(
        (bin["count"] / total) * abs(bin["accuracy"] - bin["avg_confidence"]) for bin in bins
    )


def agreement_rate(first: Sequence[str], second: Sequence[str]) -> float:
    """Fraction of samples where two labelers (or two runs) give the same label.

    Raises ``ValueError`` on a length mismatch or empty input.
    """
    if len(first) != len(second):
        raise ValueError(f"length mismatch: {len(first)} vs {len(second)} labels")
    if not first:
        raise ValueError("agreement_rate needs at least one sample")
    return sum(1 for a, b in zip(first, second) if a == b) / len(first)

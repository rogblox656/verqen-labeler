"""Tests for the conformal helpers (hand-computed expectations)."""

from __future__ import annotations

import pytest

from verqen_labeler import LabelResult
from verqen_labeler.conformal import (
    covered_fraction,
    empirical_coverage,
    extract_label_set,
    needs_review,
    set_size_stats,
)


def test_extract_label_set_from_prediction_set():
    decision = {"selected": "politics", "prediction_set": ["politics", "sports"]}
    assert extract_label_set(decision) == ["politics", "sports"]


def test_extract_label_set_falls_back_to_selected():
    assert extract_label_set({"selected": "sports"}) == ["sports"]
    assert extract_label_set({"selected": "sports", "prediction_set": []}) == ["sports"]


def test_needs_review_rules():
    assert needs_review(["a", "b"], 0.99, 0.85) is True  # set size
    assert needs_review(["a"], 0.5, 0.85) is True  # confidence below threshold
    assert needs_review(["a"], 0.85, 0.85) is False  # threshold is inclusive
    assert needs_review(["a"], 0.99, 0.85) is False


def test_set_size_stats_hand_computed():
    stats = set_size_stats([["a"], ["a", "b"], ["a", "b", "c"]])
    assert stats["count"] == 3
    assert stats["avg"] == pytest.approx(2.0)
    assert stats["min"] == 1
    assert stats["max"] == 3
    assert stats["single_rate"] == pytest.approx(1 / 3)


def test_set_size_stats_empty():
    assert set_size_stats([]) == {"count": 0, "avg": 0.0, "min": 0, "max": 0, "single_rate": 0.0}


def test_covered_fraction_hand_computed():
    sets = [["a"], ["a", "b"], ["x"]]
    truth = ["a", "b", "b"]
    assert covered_fraction(sets, truth) == pytest.approx(2 / 3)


def test_covered_fraction_errors_and_empty():
    with pytest.raises(ValueError, match="length mismatch"):
        covered_fraction([["a"]], ["a", "b"])
    assert covered_fraction([], []) == 0.0


def _result(label: str, conformal_set: list[str]) -> LabelResult:
    return LabelResult(
        text="some text",
        label=label,
        confidence=0.9,
        conformal_set=conformal_set,
        needs_human_review=False,
        latency_ms=1.0,
    )


def test_empirical_coverage_without_ground_truth():
    results = [_result("sports", ["sports"]), _result("politics", ["politics", "sports"])]
    assert empirical_coverage(results) == pytest.approx(1.0)


def test_empirical_coverage_with_ground_truth():
    results = [_result("sports", ["sports"]), _result("politics", ["politics", "sports"])]
    # only the first truth lands in its set
    assert empirical_coverage(results, ["sports", "finance"]) == pytest.approx(0.5)
    assert empirical_coverage(results, ["sports", "politics"]) == pytest.approx(1.0)

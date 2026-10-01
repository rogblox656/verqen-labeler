"""Tests for quality metrics (hand-computed expectations)."""

from __future__ import annotations

import pytest

from verqen_labeler.conformal import covered_fraction
from verqen_labeler.quality import (
    accuracy,
    agreement_rate,
    calibration_bins,
    coverage,
    expected_calibration_error,
)


def test_accuracy_hand_computed():
    assert accuracy(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(2 / 3)
    assert accuracy(["a"], ["a"]) == 1.0


def test_accuracy_errors():
    with pytest.raises(ValueError, match="length mismatch"):
        accuracy(["a"], ["a", "b"])
    with pytest.raises(ValueError, match="at least one"):
        accuracy([], [])


def test_calibration_bins_hand_computed():
    # 10 equal-width bins: 0.15 -> bin 1, 0.55 -> bin 5, 0.85 -> bin 8, 0.95 -> bin 9
    bins = calibration_bins([0.15, 0.55, 0.95, 0.85], [False, False, True, True], n_bins=10)
    assert [item["bin"] for item in bins] == [1, 5, 8, 9]
    assert all(item["count"] == 1 for item in bins)
    first = bins[0]
    assert first["lower"] == pytest.approx(0.1)
    assert first["upper"] == pytest.approx(0.2)
    assert first["avg_confidence"] == pytest.approx(0.15)
    assert first["accuracy"] == 0.0


def test_calibration_bins_top_bin_includes_one():
    bins = calibration_bins([1.0], [True], n_bins=10)
    assert bins[0]["bin"] == 9
    assert bins[0]["avg_confidence"] == 1.0


def test_ece_hand_computed():
    # each sample alone in its bin:
    # (|0 - 0.15| + |0 - 0.55| + |1 - 0.95| + |1 - 0.85|) / 4 = 0.9 / 4
    ece = expected_calibration_error([0.15, 0.55, 0.95, 0.85], [False, False, True, True])
    assert ece == pytest.approx(0.225, abs=1e-9)


def test_ece_perfectly_calibrated_is_zero():
    assert expected_calibration_error([0.5, 0.5], [True, False]) == pytest.approx(0.0)


def test_ece_two_bins_grouped():
    # bin 0: confidences 0.1, 0.2 both wrong -> |0 - 0.15| = 0.15
    # bin 1: confidences 0.8, 0.9 both right -> |1 - 0.85| = 0.15
    # ECE = 0.5 * 0.15 + 0.5 * 0.15 = 0.15
    ece = expected_calibration_error([0.9, 0.8, 0.1, 0.2], [True, True, False, False], n_bins=2)
    assert ece == pytest.approx(0.15, abs=1e-9)


def test_calibration_errors():
    with pytest.raises(ValueError, match="n_bins"):
        calibration_bins([0.5], [True], n_bins=0)
    with pytest.raises(ValueError, match="outside"):
        calibration_bins([1.5], [True])
    with pytest.raises(ValueError, match="length mismatch"):
        calibration_bins([0.5], [True, False])
    with pytest.raises(ValueError, match="at least one"):
        calibration_bins([], [])


def test_agreement_rate_hand_computed():
    assert agreement_rate(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="length mismatch"):
        agreement_rate(["a"], ["b", "c"])
    with pytest.raises(ValueError, match="at least one"):
        agreement_rate([], [])


def test_coverage_is_conformal_covered_fraction():
    assert coverage is covered_fraction
    assert coverage([["a"], ["x"]], ["a", "b"]) == pytest.approx(0.5)

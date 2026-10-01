"""Tests for active-learning selection (hand-computed expectations)."""

from __future__ import annotations

import math

import pytest

from verqen_labeler.active_learning import (
    entropy,
    normalized_entropy,
    pick_top_k,
    rank_samples,
    uncertainty_score,
)

CERTAIN = {"selected": "a", "probabilities": {"a": 1.0}, "margin": 1.0, "prediction_set": ["a"]}
COIN_FLIP = {
    "selected": "a",
    "probabilities": {"a": 0.5, "b": 0.5},
    "margin": 0.0,
    "prediction_set": ["a", "b"],
}
SKEWED = {
    "selected": "a",
    "probabilities": {"a": 0.75, "b": 0.25},
    "margin": 0.5,
    "prediction_set": ["a"],
}


def test_entropy_hand_computed():
    assert entropy({"a": 0.5, "b": 0.5}) == pytest.approx(1.0)
    assert entropy({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}) == pytest.approx(2.0)
    assert entropy({"a": 1.0}) == pytest.approx(0.0)
    assert entropy({}) == 0.0
    assert entropy({"a": 1.0, "b": 0.0}) == pytest.approx(0.0)  # zero entries skipped


def test_normalized_entropy_hand_computed():
    assert normalized_entropy({"a": 0.5, "b": 0.5}) == pytest.approx(1.0)
    assert normalized_entropy({"a": 1.0}) == 0.0
    expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert normalized_entropy({"a": 0.75, "b": 0.25}) == pytest.approx(expected)


def test_uncertainty_score_certain_decision():
    assert uncertainty_score(CERTAIN) == pytest.approx(0.0)


def test_uncertainty_score_coin_flip():
    # (normalized entropy 1.0 + margin term 1.0 + set term 1/3) / 3
    assert uncertainty_score(COIN_FLIP) == pytest.approx((1.0 + 1.0 + 1 / 3) / 3)


def test_uncertainty_score_without_optional_keys():
    # no probabilities -> 0, no margin -> 1, no prediction_set -> [selected]
    assert uncertainty_score({"selected": "a"}) == pytest.approx((0.0 + 1.0 + 0.0) / 3)


def test_rank_samples_orders_by_uncertainty():
    pool = [COIN_FLIP, CERTAIN, SKEWED]
    assert rank_samples(pool) == [0, 2, 1]
    assert pick_top_k(pool, 2) == [0, 2]


def test_pick_top_k_edges():
    pool = [CERTAIN]
    assert pick_top_k(pool, 5) == [0]
    assert pick_top_k(pool, 0) == []
    with pytest.raises(ValueError, match="k must be"):
        pick_top_k(pool, -1)

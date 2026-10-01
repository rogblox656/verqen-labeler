"""Live tests against a real VerQen checkpoint.

Skipped unless ``VERQEN_CHECKPOINT`` is set. Run with::

    pytest -m live
"""

from __future__ import annotations

import os
import time

import pytest

from verqen_labeler import Labeler

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("VERQEN_CHECKPOINT"),
        reason="VERQEN_CHECKPOINT not set",
    ),
]

CATEGORIES = ["sports", "politics", "technology", "entertainment", "finance"]


@pytest.fixture(scope="module")
def live_labeler():
    return Labeler(categories=CATEGORIES)


def test_labels_benign_text(live_labeler):
    result = live_labeler.label("The striker scored twice in the second half to seal the win.")
    assert result.label in CATEGORIES
    assert result.conformal_set
    assert set(result.conformal_set) <= set(CATEGORIES)
    assert 0.0 <= result.confidence <= 1.0
    assert result.latency_ms > 0


def test_review_flag_matches_rule(live_labeler):
    result = live_labeler.label("Investors welcomed the surprise cut in interest rates.")
    expected = (
        len(result.conformal_set) > 1
        or result.confidence < live_labeler.auto_label_threshold
    )
    assert result.needs_human_review is expected


def test_latency_budget_eager(live_labeler):
    """Warm call budget: eager mode is ~68ms on an RTX 5060 Ti; allow 2s CI slack."""
    live_labeler.label("warmup text for latency measurement")
    start = time.perf_counter()
    live_labeler.label("The committee voted to extend the program for another year.")
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < 2000


def test_label_dataset_live(live_labeler):
    results, report = live_labeler.label_dataset(
        [
            "A new film festival opens downtown this weekend.",
            "Markets rallied on stronger than expected earnings.",
        ]
    )
    assert len(results) == 2
    assert report.total == 2
    assert all(result.label in CATEGORIES for result in results)
    assert 0.0 <= report.empirical_coverage <= 1.0
    assert report.cost_usd == pytest.approx(2 * 0.046 / 1_000_000)

"""Tests for batch evaluation timing and threading."""

from __future__ import annotations

import pytest

from conftest import CATEGORIES, FakeEngine
from verqen_labeler import Labeler
from verqen_labeler.batch import evaluate_many


def test_evaluate_many_sequential(make_labeler):
    labeler = make_labeler()
    texts = ["first sample text", "second sample text", "third sample text"]

    results, timing = evaluate_many(labeler, texts)

    assert [result.text for result in results] == texts
    assert timing["count"] == 3
    assert timing["total_seconds"] >= 0.0
    assert timing["throughput_per_second"] == pytest.approx(
        3 / max(0.001, timing["total_seconds"])
    )
    assert set(timing["latency_ms"]) == {"p50", "p95", "max"}
    assert timing["latency_ms"]["p50"] <= timing["latency_ms"]["p95"] <= timing["latency_ms"]["max"]


def test_evaluate_many_threaded_preserves_order_and_counts_calls():
    engine = FakeEngine()
    labeler = Labeler(categories=CATEGORIES, engine=engine)
    texts = [f"sample number {index} with text" for index in range(12)]

    results, timing = evaluate_many(labeler, texts, num_workers=4)

    assert [result.text for result in results] == texts
    assert timing["count"] == 12
    assert len(engine.calls) == 12
    assert all(call["options"] == CATEGORIES for call in engine.calls)


def test_evaluate_many_empty(make_labeler):
    results, timing = evaluate_many(make_labeler(), [])

    assert results == []
    assert timing["count"] == 0
    assert timing["throughput_per_second"] == 0.0
    assert timing["latency_ms"] == {"p50": 0.0, "p95": 0.0, "max": 0.0}


def test_evaluate_many_invalid_workers(make_labeler):
    with pytest.raises(ValueError, match="num_workers"):
        evaluate_many(make_labeler(), ["some text"], num_workers=0)

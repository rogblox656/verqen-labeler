"""Benchmark harness tests: dataset integrity and metric math (no GPU needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "run_benchmark.py"
_spec = importlib.util.spec_from_file_location("run_benchmark", _BENCH)
run_benchmark = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_benchmark)


def test_dataset_loads_and_validates():
    rows = run_benchmark.load_dataset()
    assert len(rows) == 60
    assert {row["expected_label"] for row in rows} == set(run_benchmark.CATEGORIES)
    for category in run_benchmark.CATEGORIES:
        count = sum(1 for row in rows if row["expected_label"] == category)
        assert count == 12, f"expected 12 rows for {category}, got {count}"
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)), "duplicate dataset ids"


def test_dataset_texts_are_nonempty():
    for row in run_benchmark.load_dataset():
        assert row["text"].strip()
        assert len(row["text"]) >= 10


def test_percentile():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert run_benchmark.percentile(values, 50) == 3.0
    assert run_benchmark.percentile(values, 95) == 5.0
    assert run_benchmark.percentile([], 50) == 0.0


def test_summarize_perfect_predictions():
    records = [
        {"id": "a", "expected_label": "sports", "predicted": "sports", "correct": True,
         "auto_labeled": True, "covered": True, "set_size": 1, "confidence": 0.97,
         "latency_ms": 10.0},
        {"id": "b", "expected_label": "finance", "predicted": "finance", "correct": True,
         "auto_labeled": True, "covered": True, "set_size": 1, "confidence": 0.96,
         "latency_ms": 30.0},
    ]
    summary = run_benchmark.summarize(records)
    assert summary["accuracy"] == 1.0
    assert summary["auto_label_rate"] == 1.0
    assert summary["human_review_rate"] == 0.0
    assert summary["empirical_coverage"] == 1.0
    assert summary["avg_set_size"] == 1.0
    # nearest-rank: rank = ceil(0.5 * 2) = 1 -> first value
    assert summary["latency_ms"]["p50"] == 10.0
    assert summary["latency_ms"]["p95"] == 30.0
    assert summary["errors"] == []


def test_summarize_counts_misses():
    records = [
        {"id": "a", "expected_label": "sports", "predicted": "politics", "correct": False,
         "auto_labeled": True, "covered": True, "set_size": 2, "confidence": 0.6,
         "latency_ms": 5.0},
        {"id": "b", "expected_label": "finance", "predicted": "finance", "correct": True,
         "auto_labeled": False, "covered": True, "set_size": 2, "confidence": 0.7,
         "latency_ms": 5.0},
        {"id": "c", "expected_label": "politics", "predicted": "politics", "correct": True,
         "auto_labeled": True, "covered": False, "set_size": 1, "confidence": 0.9,
         "latency_ms": 5.0},
    ]
    summary = run_benchmark.summarize(records)
    assert summary["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["auto_label_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["human_review_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary["empirical_coverage"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["avg_set_size"] == pytest.approx(5 / 3, abs=1e-4)
    assert summary["errors"] == ["a"]


def test_summarize_empty_raises():
    with pytest.raises(ValueError):
        run_benchmark.summarize([])


def test_main_fails_cleanly_without_checkpoint(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("VERQEN_CHECKPOINT", raising=False)
    dataset = tmp_path / "tiny.jsonl"
    dataset.write_text(
        '{"id": "x1", "text": "hello there friend", "expected_label": "sports"}\n',
        encoding="utf-8",
    )
    assert run_benchmark.main(["--dataset", str(dataset)]) == 1
    assert "VERQEN_CHECKPOINT" in capsys.readouterr().err

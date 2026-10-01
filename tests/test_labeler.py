"""Tests for the Labeler class, LabelResult and LabelingReport math."""

from __future__ import annotations

import pytest

import verqen_labeler
from conftest import CATEGORIES, FakeEngine
from verqen_labeler import LabelResult, Labeler, LabelingReport


def test_package_exports_version():
    assert verqen_labeler.__version__ == "1.0.0"
    assert set(verqen_labeler.__all__) == {"Labeler", "LabelResult", "LabelingReport", "__version__"}


def test_label_returns_certified_result(make_labeler):
    labeler = make_labeler({"selected": "technology", "confidence": 0.97})
    text = "The new phone chip doubles battery life"
    result = labeler.label(text)

    assert isinstance(result, LabelResult)
    assert result.text == text
    assert result.label == "technology"
    assert result.confidence == pytest.approx(0.97)
    assert result.conformal_set == ["technology"]
    assert result.needs_human_review is False
    assert result.latency_ms >= 0.0


def test_label_passes_question_and_options(make_labeler):
    engine = FakeEngine()
    labeler = Labeler(categories=CATEGORIES, engine=engine)
    labeler.label("some sample text")

    call = engine.calls[0]
    assert call["question"] == "What is the correct label for this text?"
    assert call["options"] == CATEGORIES
    assert call["state"] == "Text: some sample text"


def test_label_passes_context_into_state():
    engine = FakeEngine()
    labeler = Labeler(categories=CATEGORIES, engine=engine)
    labeler.label("some sample text", context="sports article")

    assert engine.calls[0]["state"] == "sports article\n\nText: some sample text"


def test_label_low_confidence_flagged(make_labeler):
    result = make_labeler({"confidence": 0.5}).label("a sample piece of text")
    assert result.needs_human_review is True
    assert result.conformal_set == ["sports"]


def test_label_multi_label_set_flagged(make_labeler):
    labeler = make_labeler(
        {"selected": "politics", "confidence": 0.9, "prediction_set": ["politics", "sports"]}
    )
    result = labeler.label("election turnout reached a record high")

    assert result.conformal_set == ["politics", "sports"]
    assert result.needs_human_review is True  # set size, even though confidence is high


def test_label_empty_text_raises(make_labeler):
    with pytest.raises(ValueError, match="non-empty"):
        make_labeler().label("   ")


def test_label_dataset_report_math(make_labeler):
    labeler = make_labeler({"selected": "sports", "confidence": 0.97})
    texts = ["first sample text", "second sample text", "third sample text"]

    results, report = labeler.label_dataset(texts, labels=["sports", "sports", "politics"])

    assert isinstance(report, LabelingReport)
    assert [result.text for result in results] == texts
    assert report.total == 3
    assert report.auto_labeled == 3
    assert report.human_review_needed == 0
    assert report.estimated_accuracy == pytest.approx(0.97)
    assert report.coverage == pytest.approx(0.95)
    # only the first two ground-truth labels sit inside their conformal sets
    assert report.empirical_coverage == pytest.approx(2 / 3)
    assert report.throughput_per_second > 0.0
    assert report.cost_usd == pytest.approx(3 * 0.046 / 1_000_000)


def test_label_dataset_all_flagged_gives_zero_accuracy(make_labeler):
    labeler = make_labeler({"confidence": 0.5})
    results, report = labeler.label_dataset(["one sample text", "two sample text"])

    assert len(results) == 2
    assert report.auto_labeled == 0
    assert report.human_review_needed == 2
    assert report.estimated_accuracy == 0.0
    # no ground truth: reference is each predicted label inside its own set
    assert report.empirical_coverage == pytest.approx(1.0)


def test_label_dataset_labels_length_mismatch_raises(make_labeler):
    with pytest.raises(ValueError, match="labels must match"):
        make_labeler().label_dataset(["one sample text"], labels=["a", "b"])


def test_label_dataset_empty(make_labeler):
    results, report = make_labeler().label_dataset([])

    assert results == []
    assert report.total == 0
    assert report.auto_labeled == 0
    assert report.estimated_accuracy == 0.0
    assert report.empirical_coverage == 0.0
    assert report.throughput_per_second == 0.0
    assert report.cost_usd == 0.0


def test_invalid_categories_raise():
    with pytest.raises(ValueError, match="at least 2"):
        Labeler(categories=["only-one"], engine=FakeEngine())
    with pytest.raises(ValueError, match="distinct"):
        Labeler(categories=["a", "a"], engine=FakeEngine())
    with pytest.raises(ValueError, match="non-empty"):
        Labeler(categories=["a", "   "], engine=FakeEngine())


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError, match="coverage"):
        Labeler(categories=CATEGORIES, engine=FakeEngine(), coverage=1.5)
    with pytest.raises(ValueError, match="auto_label_threshold"):
        Labeler(categories=CATEGORIES, engine=FakeEngine(), auto_label_threshold=-0.1)


def test_missing_checkpoint_error_message(monkeypatch):
    monkeypatch.delenv("VERQEN_CHECKPOINT", raising=False)
    with pytest.raises(ValueError, match="VERQEN_CHECKPOINT"):
        Labeler(categories=CATEGORIES)

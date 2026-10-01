"""Shared test fixtures for verqen-labeler.

``FakeEngine`` mimics the real ``VerqenInferenceEngine.decide`` output shape
so the whole suite runs without a GPU, torch or a checkpoint.
"""

from __future__ import annotations

import pytest

from verqen_labeler import Labeler

CATEGORIES = ["sports", "politics", "technology", "entertainment", "finance"]


class FakeEngine:
    """Deterministic stand-in for the VerQen engine.

    Args:
        selected: Answer for the label question. ``None`` picks the first
            option; a value outside the options falls back to the first one.
        confidence: Probability assigned to the selected label (the rest is
            spread evenly over the other options).
        prediction_set: Certified set override. Pass two labels to simulate
            multi-label uncertainty; ``None`` yields ``[selected]``.
        margin: Decision margin reported in the decision dict.
    """

    def __init__(
        self,
        selected: str | None = None,
        confidence: float = 0.97,
        prediction_set: list[str] | None = None,
        margin: float = 0.9,
    ):
        self.selected = selected
        self.confidence = confidence
        self.prediction_set = prediction_set
        self.margin = margin
        self.calls: list[dict] = []

    def decide(self, state, question, options=None, question_type="choice", cost=None):
        self.calls.append(
            {
                "state": state,
                "question": question,
                "options": list(options or []),
                "type": question_type,
            }
        )
        options = list(options or [])
        if options:
            selected = self.selected if self.selected in options else options[0]
        else:
            selected = self.selected or ""
        remaining = max(0.0, 1.0 - self.confidence)
        others = remaining / max(1, len(options) - 1)
        probabilities = {option: others for option in options}
        probabilities[selected] = self.confidence
        return {
            "type": "choice",
            "selected": selected,
            "probabilities": probabilities,
            "p_correct": self.confidence,
            "margin": self.margin,
            "confidence": self.confidence,
            "should_escalate": False,
            "prediction_set": (
                list(self.prediction_set) if self.prediction_set is not None else [selected]
            ),
        }


@pytest.fixture
def fake_engine_cls():
    return FakeEngine


@pytest.fixture
def make_labeler():
    """Factory: ``make_labeler({"selected": "finance"}, coverage=0.9)``."""

    def _make(engine_kwargs: dict | None = None, **labeler_kwargs) -> Labeler:
        labeler_kwargs.setdefault("categories", list(CATEGORIES))
        return Labeler(engine=FakeEngine(**(engine_kwargs or {})), **labeler_kwargs)

    return _make

"""Core of verqen-labeler: the Labeler class and its result types."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from verqen_labeler.batch import evaluate_many
from verqen_labeler.conformal import empirical_coverage, extract_label_set, needs_review
from verqen_labeler.engine import load_engine

__all__ = ["LabelResult", "LabelingReport", "Labeler"]

_LABEL_QUESTION = "What is the correct label for this text?"
_COST_PER_MILLION = 0.046  # published VerQen rate per 1M labels (USD)


@dataclass
class LabelResult:
    """Label for a single text, with its certified conformal set.

    ``conformal_set`` is the certified label set at the labeler's target
    coverage; ``needs_human_review`` is True when that set holds more than
    one label or ``confidence`` is below the auto-label threshold.
    """

    text: str
    label: str
    confidence: float
    conformal_set: list[str]
    needs_human_review: bool
    latency_ms: float


@dataclass
class LabelingReport:
    """Aggregate quality report for a labeled batch.

    Fields:
        total: Number of samples labeled.
        auto_labeled: Samples sent straight to auto-label (no review flag).
        human_review_needed: Samples flagged for a human (multi-label set or
            confidence below the auto-label threshold).
        estimated_accuracy: Mean confidence of the auto-labeled samples
            (0.0 when nothing was auto-labeled).
        coverage: Target conformal coverage the labeler was configured with.
        empirical_coverage: Fraction of samples whose conformal set contains
            the reference label: the known ground truth when the batch was
            labeled with ``labels=...``, otherwise each sample's own
            predicted label (set integrity).
        throughput_per_second: Samples per second over the batch wall time.
        cost_usd: Estimated cost at $0.046 per 1M labels.
    """

    total: int
    auto_labeled: int
    human_review_needed: int
    estimated_accuracy: float
    coverage: float
    empirical_coverage: float
    throughput_per_second: float
    cost_usd: float


class Labeler:
    """Certified data labeler backed by the VerQen decision model.

    Args:
        categories: Label vocabulary: at least 2 distinct non-empty strings.
        checkpoint: Path to a VerQen checkpoint; falls back to the
            ``VERQEN_CHECKPOINT`` environment variable.
        coverage: Target conformal coverage for label sets (default 0.95).
        auto_label_threshold: Confidence below which a sample goes to human
            review (default 0.85).
        device: Torch device for the engine (``"auto"`` picks GPU when present).
        fast: Compile the GPU path for lower latency (see VerQen model card).
        engine: Inject a pre-built engine or test double exposing
            ``decide(state, question, options)``.
    """

    def __init__(
        self,
        categories: Sequence[str],
        checkpoint: str | None = None,
        coverage: float = 0.95,
        auto_label_threshold: float = 0.85,
        device: str = "auto",
        *,
        fast: bool = False,
        engine: Any | None = None,
    ):
        labels = [str(category) for category in categories]
        if len(labels) < 2:
            raise ValueError("categories needs at least 2 labels")
        if any(not label.strip() for label in labels):
            raise ValueError("categories must be non-empty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("categories must be distinct")
        if not 0.0 < coverage <= 1.0:
            raise ValueError("coverage must be in (0, 1]")
        if not 0.0 <= auto_label_threshold <= 1.0:
            raise ValueError("auto_label_threshold must be in [0, 1]")

        self.categories = labels
        self.coverage = float(coverage)
        self.auto_label_threshold = float(auto_label_threshold)
        self._engine = engine if engine is not None else load_engine(checkpoint, device, fast=fast)
        self._lock = threading.Lock()

    def label(self, text: str, context: str = "") -> LabelResult:
        """Label one text and return a :class:`LabelResult`.

        Raises ``ValueError`` for empty input. Safe to call concurrently:
        all engine calls share one lock.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        start = time.perf_counter()
        state = f"{context}\n\nText: {text}" if context else f"Text: {text}"
        with self._lock:
            decision = self._engine.decide(state, _LABEL_QUESTION, self.categories)
        latency_ms = (time.perf_counter() - start) * 1000

        selected = str(decision.get("selected", ""))
        probabilities = decision.get("probabilities") or {}
        confidence = float(probabilities.get(selected, decision.get("confidence", 0.0)))
        conformal_set = extract_label_set(decision)
        return LabelResult(
            text=text,
            label=selected,
            confidence=confidence,
            conformal_set=conformal_set,
            needs_human_review=needs_review(conformal_set, confidence, self.auto_label_threshold),
            latency_ms=latency_ms,
        )

    def label_dataset(
        self,
        texts: Sequence[str],
        context: str = "",
        labels: Sequence[str] | None = None,
        num_workers: int = 1,
    ) -> tuple[list[LabelResult], LabelingReport]:
        """Label a whole dataset and return ``(results, report)``.

        *labels* are optional known ground-truth labels (one per text); when
        given, the report's ``empirical_coverage`` measures how often the
        true label lands inside its certified conformal set.
        """
        items = list(texts)
        if labels is not None and len(labels) != len(items):
            raise ValueError("labels must match texts in length")

        results, timing = evaluate_many(self, items, context=context, num_workers=num_workers)
        auto_labeled = [result for result in results if not result.needs_human_review]
        total = len(results)
        if auto_labeled:
            estimated_accuracy = sum(result.confidence for result in auto_labeled) / len(
                auto_labeled
            )
        else:
            estimated_accuracy = 0.0
        report = LabelingReport(
            total=total,
            auto_labeled=len(auto_labeled),
            human_review_needed=total - len(auto_labeled),
            estimated_accuracy=estimated_accuracy,
            coverage=self.coverage,
            empirical_coverage=empirical_coverage(results, labels),
            throughput_per_second=timing["throughput_per_second"],
            cost_usd=total * _COST_PER_MILLION / 1_000_000,
        )
        return results, report

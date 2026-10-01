"""Pydantic v2 request/response schemas for the REST API."""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    "LabelRequest",
    "LabelResponse",
    "LabelDatasetRequest",
    "LabelingReportModel",
    "LabelDatasetResponse",
    "HealthResponse",
]


class LabelRequest(BaseModel):
    """POST /label body: one text to label."""

    text: str = Field(min_length=1, description="Text to label")
    context: str = Field(default="", description="Optional context prepended to the state")


class LabelResponse(BaseModel):
    """One certified label (mirrors verqen_labeler.LabelResult)."""

    text: str
    label: str
    confidence: float
    conformal_set: list[str]
    needs_human_review: bool
    latency_ms: float


class LabelDatasetRequest(BaseModel):
    """POST /label_dataset body: a batch of texts."""

    texts: list[str] = Field(min_length=1)
    context: str = ""


class LabelingReportModel(BaseModel):
    """Batch quality report (mirrors verqen_labeler.LabelingReport)."""

    total: int
    auto_labeled: int
    human_review_needed: int
    estimated_accuracy: float
    coverage: float
    empirical_coverage: float
    throughput_per_second: float
    cost_usd: float


class LabelDatasetResponse(BaseModel):
    """POST /label_dataset response."""

    results: list[LabelResponse]
    report: LabelingReportModel


class HealthResponse(BaseModel):
    """GET /healthz response."""

    status: str
    categories: list[str]

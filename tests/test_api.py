"""Tests for the FastAPI app factory."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import CATEGORIES, FakeEngine
from verqen_labeler import Labeler
from verqen_labeler.api import create_api


def _client(**engine_kwargs) -> TestClient:
    labeler = Labeler(categories=CATEGORIES, engine=FakeEngine(**engine_kwargs))
    return TestClient(create_api(labeler=labeler))


def test_healthz():
    response = _client().get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["categories"] == CATEGORIES


def test_label_endpoint():
    client = _client(selected="finance", confidence=0.99)
    response = client.post("/label", json={"text": "the central bank raised rates again"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "the central bank raised rates again"
    assert payload["label"] == "finance"
    assert payload["confidence"] == pytest.approx(0.99)
    assert payload["conformal_set"] == ["finance"]
    assert payload["needs_human_review"] is False
    assert payload["latency_ms"] >= 0.0


def test_label_with_context():
    client = _client()
    response = client.post("/label", json={"text": "sample text here", "context": "sports news"})
    assert response.status_code == 200


def test_label_empty_text_rejected_by_schema():
    assert _client().post("/label", json={"text": ""}).status_code == 422


def test_label_whitespace_text_returns_400():
    response = _client().post("/label", json={"text": "   "})
    assert response.status_code == 400
    assert "non-empty" in response.json()["detail"]


def test_label_dataset_endpoint():
    client = _client(selected="sports", confidence=0.97)
    response = client.post("/label_dataset", json={"texts": ["first sample", "second sample"]})

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["results"]) == 2
    assert payload["results"][0]["label"] == "sports"
    report = payload["report"]
    assert report["total"] == 2
    assert report["auto_labeled"] == 2
    assert report["human_review_needed"] == 0
    assert report["coverage"] == pytest.approx(0.95)
    assert report["empirical_coverage"] == pytest.approx(1.0)
    assert report["cost_usd"] == pytest.approx(2 * 0.046 / 1_000_000)


def test_label_dataset_empty_rejected():
    assert _client().post("/label_dataset", json={"texts": []}).status_code == 422


def test_create_api_requires_labeler_or_categories(monkeypatch):
    monkeypatch.delenv("VERQEN_CHECKPOINT", raising=False)
    with pytest.raises(ValueError, match="labeler"):
        create_api()

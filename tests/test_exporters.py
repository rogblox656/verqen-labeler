"""Tests for the exporters registry and each format's round trip."""

from __future__ import annotations

import csv
import json

import pytest

from verqen_labeler import LabelResult
from verqen_labeler.exporters import available_formats, export


def _results() -> list[LabelResult]:
    return [
        LabelResult(
            text="the team won the final",
            label="sports",
            confidence=0.97,
            conformal_set=["sports"],
            needs_human_review=False,
            latency_ms=1.5,
        ),
        LabelResult(
            text="rates rose again",
            label="finance",
            confidence=0.6,
            conformal_set=["finance", "politics"],
            needs_human_review=True,
            latency_ms=2.25,
        ),
    ]


def test_available_formats():
    assert available_formats() == ["coco", "csv", "hf", "label_studio"]


def test_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown export format"):
        export(_results(), tmp_path / "out", "xml")


def test_csv_roundtrip(tmp_path):
    path = export(_results(), tmp_path / "labels.csv", "csv")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 2
    assert rows[0]["text"] == "the team won the final"
    assert rows[0]["label"] == "sports"
    assert float(rows[0]["confidence"]) == pytest.approx(0.97)
    assert json.loads(rows[0]["conformal_set"]) == ["sports"]
    assert rows[0]["needs_human_review"] == "false"
    assert json.loads(rows[1]["conformal_set"]) == ["finance", "politics"]
    assert rows[1]["needs_human_review"] == "true"


def test_hf_jsonl_roundtrip(tmp_path):
    path = export(_results(), tmp_path / "labels.jsonl", "hf")

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["label"] == "sports"
    assert records[0]["conformal_set"] == ["sports"]
    assert records[1]["needs_human_review"] is True
    assert records[1]["confidence"] == pytest.approx(0.6)


def test_coco_structure(tmp_path):
    path = export(_results(), tmp_path / "coco.json", "coco")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert set(payload) == {"info", "images", "annotations", "categories"}
    assert len(payload["images"]) == 2
    assert payload["images"][0]["text"] == "the team won the final"
    assert payload["images"][0]["file_name"] == "text-0001.txt"
    assert len(payload["annotations"]) == 2
    assert payload["annotations"][0]["category_id"] == 1
    assert payload["annotations"][1]["category_id"] == 2
    assert payload["annotations"][0]["bbox"] == [0, 0, 0, 0]
    assert [category["name"] for category in payload["categories"]] == ["sports", "finance"]


def test_label_studio_structure(tmp_path):
    path = export(_results(), tmp_path / "ls.json", "label_studio")
    tasks = json.loads(path.read_text(encoding="utf-8"))

    assert len(tasks) == 2
    task = tasks[0]
    assert task["data"]["text"] == "the team won the final"
    prediction = task["predictions"][0]
    assert prediction["model_version"] == "verqen"
    assert prediction["score"] == pytest.approx(0.97)
    result = prediction["result"][0]
    assert result["type"] == "labels"
    assert result["value"]["labels"] == ["sports"]
    assert result["value"]["start"] == 0
    assert result["value"]["end"] == len("the team won the final")
    # the second task carries the full certified set
    assert tasks[1]["predictions"][0]["result"][0]["value"]["labels"] == ["finance", "politics"]


def test_export_creates_parent_dirs(tmp_path):
    path = export(_results(), tmp_path / "deep" / "nested" / "labels.csv", "csv")
    assert path.exists()


def test_empty_export_writes_valid_files(tmp_path):
    csv_path = export([], tmp_path / "empty.csv", "csv")
    hf_path = export([], tmp_path / "empty.jsonl", "hf")
    coco_path = export([], tmp_path / "empty_coco.json", "coco")
    ls_path = export([], tmp_path / "empty_ls.json", "label_studio")

    # CSV has only the header row
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    # HF JSONL has no lines at all
    assert hf_path.read_text(encoding="utf-8") == ""
    coco_payload = json.loads(coco_path.read_text(encoding="utf-8"))
    assert coco_payload["images"] == []
    assert coco_payload["annotations"] == []
    assert coco_payload["categories"] == []
    assert json.loads(ls_path.read_text(encoding="utf-8")) == []

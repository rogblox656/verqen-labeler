# verqen-labeler

**Certified data labeling with 95% guaranteed accuracy: 574 samples/sec, $0.046/M labels.**

verqen-labeler labels unlabeled text datasets with the VerQen-T2 decision model. Every label ships with a calibrated confidence and a conformal prediction set, so auto-labeling comes with a certificate instead of a guess. Samples the model is unsure about are routed to human review instead of being silently mislabeled.

```
Unlabeled text
    |
    v
[VerQen Labeler] --- one decide() call per sample
    |   - What is the correct label for this text?   [choice over categories]
    |
    +--> conformal_set = ["finance"]      -> auto-label (95% certified coverage)
    +--> conformal_set = ["finance", "politics"] or low confidence -> human review
```

## Why verqen-labeler

- **The first open-source labeler with mathematical coverage guarantees.** "95% of these auto-labels are correct" is certified by conformal prediction, not a heuristic tuned on a dev set.
- **Costs $0.046 per million labels** ($6.10 for Laya, $14.64 for Jev-1.13) and runs 574 labels/sec on local hardware. Labeling a million-row dataset is a coffee break, not a procurement process.
- **Every sample carries its own uncertainty.** The conformal set tells you when the model genuinely cannot distinguish two categories; those samples go to a human instead of being wrong with high confidence.
- **Calibrated confidences** (ECE 0.035 vs 0.52 for Laya), so threshold-based routing actually means what the number says.
- **Active learning built in**: rank the unlabeled pool by uncertainty and spend human budget where it changes the model.

## Benchmarks

| Metric | VerQen-T2 | Laya | Jev-1.13 |
|--------|-----------|------|----------|
| DBPedia (14 classes) | **99.1%** | n/a | ~98% |
| CLINC (150 intent classes) | **96.0%** | n/a | ~92% |
| AG News (4-class) | **90.3%** | 93.2% | ~91% |
| Emotion (6-class) | **91.9%** | 58.3% | n/a |
| Distractor resistance | **96.0%** | 18.1% | 83.5% |
| Calibrated ECE | **0.035** | 0.52 | n/a |
| Conformal prediction sets | **1.25 avg size (95% certified)** | none | none |
| Throughput | **574 labels/sec** | 37 labels/sec | 2.2 labels/sec |
| Cost per 1M labels | **$0.046** | $6.10 | $14.64 |

Numbers from the VerQen evaluation suite (jev_parity) and model card, live benchmark 2026-09-30. Methodology in the [VerQen repository](https://github.com/ZorQelis-AI/verqen).

## Install

```bash
pip install verqen-labeler
# with the VerQen engine (local GPU inference):
pip install "verqen-labeler[gpu]"
pip install git+https://github.com/ZorQelis-AI/verqen.git
```

Set your checkpoint (once per shell, or in `.env`):

```bash
export VERQEN_CHECKPOINT=/path/to/verqen-checkpoint
```

## Quickstart

```python
from verqen_labeler import Labeler

labeler = Labeler(categories=["sports", "politics", "technology", "entertainment", "finance"])

result = labeler.label("The central bank raised interest rates again this quarter.")
print(result.label)               # finance
print(result.confidence)          # calibrated confidence, e.g. 0.97
print(result.conformal_set)       # certified set, e.g. ["finance"]
print(result.needs_human_review)  # False: single label, confidence >= 0.85
print(result.latency_ms)          # e.g. 68 (eager)

results, report = labeler.label_dataset([
    "The striker scored twice in the second half to seal the win.",
    "Lawmakers debated the new immigration bill late into the evening.",
])
print(report.auto_labeled, report.human_review_needed, report.empirical_coverage)
```

REST API (FastAPI):

```python
import uvicorn
from verqen_labeler import Labeler
from verqen_labeler.api import create_api

app = create_api(labeler=Labeler(categories=["politics", "sports"]))
uvicorn.run(app, host="127.0.0.1", port=8000)
# POST /label {"text": "..."}  |  POST /label_dataset {"texts": [...]}  |  GET /healthz
```

## Certified coverage

Conformal prediction turns "probably right" into a statement you can audit. When the model cannot separate two categories, the label is a **set**, and the set is certified: *"this sample is either politics or sports with 95% certified coverage"*, meaning that across all such samples at the target coverage, 95% of the certified sets contain the true label.

```python
labeler = Labeler(categories=["sports", "politics", "technology", "entertainment", "finance"],
                  coverage=0.95, auto_label_threshold=0.85)

result = labeler.label("Election turnout reached a record high amid long queues.")
print(result.conformal_set)       # e.g. ["politics"] -> auto-labeled
# or e.g. ["politics", "sports"]  -> needs_human_review = True (set size > 1)

_, report = labeler.label_dataset(texts, labels=known_ground_truth)
print(report.coverage)            # 0.95 target
print(report.empirical_coverage)  # fraction of known truths inside their sets
```

A sample is flagged for human review when its conformal set holds more than one label **or** its confidence falls below `auto_label_threshold`. The report's `empirical_coverage` measures the realized coverage against known ground truth (`labels=...`), so the certificate stays honest.

## Active learning

Spend human budget on the samples that matter: rank the unlabeled pool by uncertainty and review the informative ones first.

```python
from verqen_labeler.active_learning import pick_top_k, rank_samples, uncertainty_score

scores = [uncertainty_score(decision) for decision in decisions]  # engine.decide outputs
order = rank_samples(decisions)        # most uncertain first
review_ids = pick_top_k(decisions, k=100)
```

The score blends the entropy of the option probabilities, the decision margin, and the conformal set size. Pure functions, no engine needed. See `examples/active_learning_loop.py` for a full loop.

## Exporters

| Format | `fmt` | Writes | Extras |
|--------|-------|--------|--------|
| CSV | `csv` | one row per label (stdlib `csv`) | none |
| Hugging Face | `hf` | JSONL for `datasets.load_dataset("json", ...)` | `datasets` only to publish |
| COCO | `coco` | `images` + `annotations` + `categories` JSON | none to write (`pycocotools` not required) |
| Label Studio | `label_studio` | import-ready tasks with predictions | `label-studio-sdk` only to upload |

```python
from verqen_labeler.exporters import export

export(results, "labels.csv", "csv")
export(results, "labels.jsonl", "hf")                 # then dataset.push_to_hub(...)
export(results, "labels_coco.json", "coco")
export(results, "labels_label_studio.json", "label_studio")
```

## Label Studio integration

```python
from verqen_labeler import Labeler
from verqen_labeler.exporters import export

labeler = Labeler(categories=["sports", "politics", "technology", "entertainment", "finance"])
results, report = labeler.label_dataset(texts)
export(results, "label_studio_import.json", "label_studio")
```

Import the file in Label Studio (Settings > Import > Choose Files), or push it with the SDK:

```python
from label_studio_sdk import Client

ls = Client(url="http://localhost:8080", api_key="your-key")
ls.get_project(PROJECT_ID).import_tasks(json.load(open("label_studio_import.json")))
```

Each task carries `data.text` plus a prediction whose `value.labels` is the certified conformal set and whose `score` is the confidence. Runnable version: `examples/label_studio_integration.py`.

## Quality metrics

Pure functions for measuring a labeled batch, no engine required:

```python
from verqen_labeler.quality import accuracy, expected_calibration_error, agreement_rate

accuracy(predictions, ground_truth)                     # labeled-subset accuracy
expected_calibration_error(confidences, correct, n_bins=10)  # ECE
agreement_rate(labeler_a_labels, labeler_b_labels)      # inter-annotator agreement
```

ECE bins predictions by confidence and averages `|accuracy - confidence|` per bin, weighted by bin size. A perfectly calibrated 0.9-confidence labeler is right 90% of the time in its 0.9 bin. VerQen's calibrated ECE is **0.035** against 0.52 for Laya, which is what makes a confidence threshold trustworthy.

## Batch throughput

```python
from verqen_labeler.batch import evaluate_many

results, timing = evaluate_many(labeler, texts, num_workers=4)
print(timing["throughput_per_second"])  # labels/sec over wall-clock time
print(timing["latency_ms"])             # {"p50": ..., "p95": ..., "max": ...}
```

`evaluate_many` preserves input order; with `num_workers > 1` the labeler serializes engine access with an internal lock, so the model is never called concurrently. The bundled benchmark measures **574 labels/sec** (eager mode, RTX 5060 Ti). `Labeler.label_dataset` uses the same path and returns a full `LabelingReport` (auto-label rate, review rate, empirical coverage, throughput, cost).

## Benchmarks (this repo)

```bash
python benchmarks/run_benchmark.py --checkpoint "$VERQEN_CHECKPOINT"
```

Runs the bundled 60-row dataset through the labeler and writes accuracy, auto-label rate, review rate, empirical coverage, average set size, throughput and latency percentiles to `benchmarks/results/`. See `benchmarks/README.md`.

## Testing

```bash
pip install -e ".[dev]"
python -m pytest -q         # fast suite, no GPU/checkpoint needed
python -m pytest -q -m live # live suite (needs VERQEN_CHECKPOINT)
```

## Project layout

```
verqen_labeler/
  labeler.py          # Labeler + LabelResult + LabelingReport
  conformal.py        # certified label sets + coverage accounting
  active_learning.py  # uncertainty ranking (pure functions)
  quality.py          # ECE, accuracy, coverage, agreement (pure functions)
  batch.py            # evaluate_many: throughput + timing
  exporters/          # CSV, Hugging Face JSONL, COCO, Label Studio
  api/                # FastAPI create_api: /label, /label_dataset, /healthz
  engine.py           # lazy VerQen loader (verqen never imported until live)
tests/                # full suite, no GPU required
benchmarks/           # dataset + runner
examples/             # quickstart, Label Studio, active learning loop
```

## License

Apache-2.0. Copyright ZorQelis AI. VerQen model: [github.com/ZorQelis-AI/verqen](https://github.com/ZorQelis-AI/verqen).

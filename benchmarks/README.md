# Benchmark suite

Runs the bundled `labeling_dataset.jsonl` through a live `verqen-labeler`.

```bash
export VERQEN_CHECKPOINT=/path/to/verqen-checkpoint
python benchmarks/run_benchmark.py
python benchmarks/run_benchmark.py --limit 20
python benchmarks/run_benchmark.py --fail-under 0.9   # CI gate, exit 3 on failure
```

## Dataset

Starter regression suite: 60 rows, 12 per topic (`sports`, `politics`,
`technology`, `entertainment`, `finance`). Each row:

```json
{"id": "lbl-001", "text": "...", "expected_label": "sports"}
```

> The official accuracy numbers in the README (99.1% DBPedia, 96.0% CLINC,
> 0.035 calibrated ECE) come from the full VerQen evaluation suite
> (`jev_parity`) and model card, live benchmark 2026-09-30, not from this
> smaller starter set. Append your own cases to `labeling_dataset.jsonl`;
> `run_benchmark.py --dataset path/to/your.jsonl` accepts any file with the
> same schema.

## Results

Reports are written to `results/benchmark-<timestamp>.json` (the `results/`
directory is created on first run). Fields: `metrics.accuracy`,
`auto_label_rate`, `human_review_rate`, `empirical_coverage`
(expected label inside the conformal set), `avg_set_size`,
`throughput_per_second`, `latency_ms.p50/p95/max`, `errors` (ids of missed
rows), plus every per-row record.

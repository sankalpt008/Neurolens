# NeuroLens

NeuroLens unifies GPU profiling runs into a strict, shareable ledger. The project now spans the Phase 0 groundwork (docs, schema, validation) and Phase 1–2 runtime adapters for ONNX Runtime, PyTorch, and TensorRT (with graceful stub).

## Quickstart

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
# Optional backends
pip install -e .[profiler]        # ONNX Runtime helper
pip install onnx torch            # Install manually if extras unavailable
```

### 2. Run tests
```bash
pytest -q
```
Tests automatically skip adapters whose dependencies are missing (e.g., TensorRT or PyTorch).

### 3. Profile models via CLI
```bash
# ONNX Runtime
python tools/gen_add_onnx.py --out /tmp/add.onnx
neurolens profile --backend onnxrt --model /tmp/add.onnx

# PyTorch builtin tiny model (requires torch installed)
python tools/gen_tiny_linear.py --out-dir /tmp/tiny
neurolens profile --backend torch --model builtin:tiny_linear

# TensorRT (requires generated ONNX; stub fallback if TRT unavailable)
neurolens profile --backend tensorrt --model /tmp/tiny/tiny_linear.onnx --allow-stub-trt
```
The command writes a schema-compliant artifact to `runs/` and prints a latency summary. Use `neurolens profile --help` for the full CLI reference.

### 4. Fingerprints & Diff
```bash
# Build a fingerprint from a run artifact
neurolens fingerprint --run runs/example.json --out fp/example.fp.json

# Compare two fingerprints and export a Markdown report
neurolens compare --a fp/base.fp.json --b fp/latest.fp.json --topk 10 --markdown out/compare.md
```
Fingerprints capture per-op vectors normalized by run totals and optional hardware peaks. Similarity uses cosine distance across
aligned op signatures, and the diff output highlights which layers shifted most.

### 5. Generate Insights & Reports
```bash
# From a validated profiling run
neurolens report --run runs/example.json --md out/report.md --html out/report.html

# From fingerprints with a baseline
neurolens compare --a fp/base.fp.json --b fp/latest.fp.json --markdown out/diff.md
neurolens report --fingerprint fp/latest.fp.json --baseline fp/base.fp.json --html out/report.html
```
Insights evaluate the rule DSL in `neurolens/insights/rules.yaml`, rank findings by severity and impact, and emit Markdown/HTML summaries that highlight global issues, per-op bottlenecks, and divergences versus an optional baseline.

### 6. Local Visualization
```bash
# Launch the dashboard with a profiling run preloaded
neurolens view --run runs/example.json

# Explore a fingerprint directly
neurolens view --fingerprint fp/example.fp.json

# Load a baseline/candidate pair and open the diff tab
neurolens view --compare fp/base.fp.json fp/latest.fp.json
```
The Streamlit-powered dashboard offers timeline, roofline, and fingerprint diff pages. All data is loaded from local JSON files so the experience remains fully offline.

### 7. Export Bundles
```bash
# Generate artifacts
neurolens fingerprint --run runs/run123.json --out fp/run123.fp.json
neurolens report --run runs/run123.json --html reports/run123.html

# Package everything into a bundle
neurolens export --run runs/run123.json --fingerprint fp/run123.fp.json \
  --report reports/run123.html --out-dir exports
```
The resulting `.bundle.zip` archive collects the run, fingerprint, reports, and a `manifest.json` with SHA-256 hashes for integrity checks.

### 8. Validate a profiling JSON manually
```bash
python - <<'PY'
from pathlib import Path
import json
from neurolens.utils.validate import validate_run_schema

data = json.load(open('samples/trace_minimal.json'))
validate_run_schema(data)
print('Validation succeeded!')
PY
```

## Project Layout
- `docs/` — specifications, metrics glossary, and architecture notes.
- `schema/` — JSON schema definitions for profiling runs.
- `neurolens/core/` — profiling orchestrator and run writers.
- `neurolens/adapters/` — backend-specific adapters and registry.
- `neurolens/fingerprint/` — fingerprint builder and similarity utilities.
- `neurolens/utils/` — environment detection, schema validation, and JSON helpers.
- `neurolens/cli/` — Typer-powered CLI entrypoints.
- `samples/` — example traces; generate models via `tools/` helpers when needed.
- `golden/` — canonical traces for parity checks.
- `tests/` — Pytest-based validation harness for schema and adapters.
- `devlog/` — Daily development journal capturing progress and next steps.

## License
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

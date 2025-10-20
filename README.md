# NeuroLens

[![CI](https://github.com/neurolens/neurolens/actions/workflows/ci.yml/badge.svg)](https://github.com/neurolens/neurolens/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/neurolens.svg)](https://pypi.org/project/neurolens/)

NeuroLens normalises GPU profiling runs into a strict schema, builds per-layer
fingerprints, and surfaces actionable insights through reports and dashboards. The tool
chain targets ML performance engineers who need consistent telemetry across ONNX
Runtime, PyTorch, and TensorRT workloads.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
```

Optional extras:

- `pip install -e .[profiler]` – install adapter backends (ONNX Runtime, PyTorch)
- `pip install -e .[viz]` – enable the Streamlit dashboard

## CLI Overview

`neurolens --help` lists every command. Highlights:

| Command | Purpose |
| --- | --- |
| `neurolens profile` | Run a model through the selected backend with profiling enabled |
| `neurolens ingest` | Validate existing artifacts and append them to the run store |
| `neurolens fingerprint` | Convert run JSON into normalised vector fingerprints |
| `neurolens compare` | Align fingerprints, compute similarity, and report divergences |
| `neurolens report` | Evaluate rules and emit Markdown/HTML insight reports |
| `neurolens view` | Launch the local Streamlit dashboard (timeline, roofline, diff) |
| `neurolens bench` | Sweep batch/seq/precision grids and save per-run artifacts |
| `neurolens export` | Bundle runs, fingerprints, and reports into a ZIP archive |
| `neurolens ls` | Query the manifest/index-backed run store |

Each subcommand provides detailed `--help` text, parameter descriptions, and graceful
error messages for missing files or schema violations.

## Quickstart

Follow the step-by-step walkthrough in [`docs/quickstart.md`](docs/quickstart.md) to
generate a toy ONNX model, profile it, build fingerprints, and launch the dashboard.

## Documentation

- [`docs/architecture_overview.md`](docs/architecture_overview.md) – data flow and module layout
- [`docs/metrics_glossary.md`](docs/metrics_glossary.md) – counter definitions and units
- [`docs/rules_guide.md`](docs/rules_guide.md) – rule DSL and insights engine reference
- [`docs/cookbook.md`](docs/cookbook.md) – interpreting bottlenecks and metrics
- [`docs/storage_guide.md`](docs/storage_guide.md) – manifest/index storage design
- [`docs/export_guide.md`](docs/export_guide.md) – bundling artifacts for sharing

## Development

```bash
pip install -e .[dev]
ruff neurolens tests
pytest -q
```

Tests auto-skip adapters when optional dependencies (e.g., TensorRT) are unavailable.
`docs/bench_guide.md` and `docs/visualization_guide.md` cover advanced workflows.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

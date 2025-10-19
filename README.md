# NeuroLens

NeuroLens unifies GPU profiling runs into a strict, shareable ledger. The project now includes the Phase 0 groundwork (docs, schema, validation) and the Phase 1 ONNX Runtime profiling pipeline.

## Quickstart

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

> **Note:** To execute real ONNX Runtime profiles install the optional extras: `pip install -e .[profiler]`.

### 2. Run tests
```bash
pytest
```

### 3. Profile an ONNX model
```bash
neurolens profile --model path/to/model.onnx --bs 8 --seq 128 --precision fp16
```
The command writes a schema-compliant artifact to `runs/` and prints a latency summary. Add `--help` for the full CLI reference.

### 4. Validate a profiling JSON manually
```bash
python - <<'PY'
from pathlib import Path
import json
import jsonschema

schema = json.load(open('schema/run.schema.json'))
data = json.load(open('samples/trace_minimal.json'))
jsonschema.validate(instance=data, schema=schema)
print('Validation succeeded!')
PY
```

## Project Layout
- `docs/` — specifications, metrics glossary, and architecture notes.
- `schema/` — JSON schema definitions for profiling runs.
- `neurolens/core/` — profiling orchestrator and schema validation helpers.
- `neurolens/adapters/` — backend-specific adapters (Phase 1: ONNX Runtime).
- `neurolens/cli/` — Typer-powered CLI entrypoints.
- `samples/` — Example traces that validate against the schema.
- `tests/` — Pytest-based validation harness for schema and profiler logic.
- `devlog/` — Daily development journal capturing progress and next steps.

## License
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

# NeuroLens

NeuroLens unifies GPU profiling runs into a strict, shareable ledger. This repository currently hosts the Phase 0 groundwork: schema, docs, and validation tests.

## Quickstart

### 1. Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

### 2. Run tests
```bash
pytest
```

### 3. Validate a profiling JSON
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
- `samples/` — Example traces that validate against the schema.
- `tests/` — Pytest-based validation harness.
- `devlog/` — Daily development journal capturing progress and next steps.

## License
Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).

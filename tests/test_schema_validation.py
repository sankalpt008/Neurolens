import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import jsonschema  # noqa: E402


SCHEMA_PATH = PROJECT_ROOT / "schema" / "run.schema.json"
SAMPLE_PATH = PROJECT_ROOT / "samples" / "trace_minimal.json"


def load_schema():
    with SCHEMA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_sample():
    with SAMPLE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_sample_trace_validates():
    schema = load_schema()
    sample = load_sample()
    jsonschema.validate(instance=sample, schema=schema)


def test_invalid_trace_rejected():
    schema = load_schema()
    sample = load_sample()
    bad = copy.deepcopy(sample)
    bad["hardware"]["extra_field"] = "not allowed"
    try:
        jsonschema.validate(instance=bad, schema=schema)
    except jsonschema.exceptions.ValidationError:
        return
    raise AssertionError("Invalid trace with extra field should fail schema validation")

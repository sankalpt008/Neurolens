"""Schema validation helpers."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import jsonschema

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schema" / "run.schema.json"
Draft7Validator = getattr(jsonschema, "Draft7Validator", None)
ValidationError = getattr(jsonschema, "exceptions", jsonschema).__dict__.get("ValidationError", Exception)


class SchemaValidationError(RuntimeError):
    """Raised when a run payload fails JSON Schema validation."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@lru_cache(maxsize=1)
def _load_schema() -> Dict[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _build_validator():
    schema = _load_schema()
    if Draft7Validator is None:  # pragma: no cover - executed when draft7 class missing
        return None
    return Draft7Validator(schema)


def validate_run_schema(run_dict: Dict[str, Any]) -> None:
    """Validate ``run_dict`` or raise :class:`SchemaValidationError`."""

    validator = _build_validator()
    if validator is not None:
        errors = sorted(validator.iter_errors(run_dict), key=lambda err: err.path)
        if errors:
            formatted = "\n".join(
                f"{'/'.join(str(x) for x in error.path)}: {error.message}".strip() or error.message
                for error in errors
            )
            raise SchemaValidationError(formatted)
        return

    try:
        jsonschema.validate(instance=run_dict, schema=_load_schema())
    except ValidationError as exc:  # pragma: no cover - depends on fallback path
        raise SchemaValidationError(str(exc)) from exc

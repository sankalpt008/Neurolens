"""Core profiling orchestration for NeuroLens."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import jsonschema

try:  # pragma: no cover - optional dependency branch
    from jsonschema import Draft7Validator
except ImportError:  # pragma: no cover
    Draft7Validator = None  # type: ignore[assignment]

from jsonschema import ValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "schema" / "run.schema.json"


class SchemaValidationError(RuntimeError):
    """Raised when a run artifact fails schema validation."""

    def __init__(self, errors: str) -> None:
        super().__init__(errors)
        self.errors = errors


_schema_cache: Optional[Dict[str, Any]] = None
_validator_cache: Optional[Any] = None


def _load_schema() -> Dict[str, Any]:
    global _schema_cache
    if _schema_cache is None:
        with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
            _schema_cache = json.load(fh)
    return _schema_cache


def _get_validator() -> Optional[Any]:
    global _validator_cache
    if Draft7Validator is None:
        return None
    if _validator_cache is None:
        _validator_cache = Draft7Validator(_load_schema())
    return _validator_cache


def validate_run_schema(run_dict: Dict[str, Any]) -> bool:
    """Validate ``run_dict`` against ``schema/run.schema.json``."""

    validator = _get_validator()
    if validator is not None:
        errors = sorted(validator.iter_errors(run_dict), key=lambda err: err.path)
        if errors:
            formatted = "\n".join(
                f"{'/'.join(str(x) for x in err.path)}: {err.message}".strip()
                for err in errors
            )
            raise SchemaValidationError(formatted)
        return True

    try:
        jsonschema.validate(instance=run_dict, schema=_load_schema())
    except ValidationError as exc:  # pragma: no cover - executed when Draft7Validator unavailable
        raise SchemaValidationError(str(exc)) from exc
    return True


@dataclass
class ProfilerResult:
    """Structured response when profiling completes."""

    run_dict: Dict[str, Any]
    output_path: Path


class Profiler:
    """Coordinates adapter execution and artifact persistence."""

    def __init__(
        self,
        adapter: Any,
        output_dir: Path | str | None = None,
        schema_path: Path | None = None,
    ) -> None:
        self.adapter = adapter
        self.output_dir = Path(output_dir) if output_dir else REPO_ROOT / "runs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.schema_path = schema_path or SCHEMA_PATH

    def profile(
        self,
        model_path: Path | str,
        *,
        batch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        precision: str = "fp32",
    ) -> ProfilerResult:
        """Execute the adapter and persist the validated run artifact."""

        base_payload = self.adapter.profile_model(
            Path(model_path),
            batch_size=batch_size,
            sequence_length=sequence_length,
            precision=precision,
        )

        run_payload: Dict[str, Any] = {
            "run_id": str(uuid.uuid4()),
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        run_payload.update(base_payload)

        validate_run_schema(run_payload)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = self.output_dir / f"run_{timestamp}.json"
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(run_payload, fh, indent=2, sort_keys=False)

        return ProfilerResult(run_dict=run_payload, output_path=output_path)


__all__ = ["Profiler", "ProfilerResult", "SchemaValidationError", "validate_run_schema"]

"""Core profiling orchestration for NeuroLens."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from neurolens.adapters import get_adapter
from neurolens.utils.validate import SchemaValidationError, validate_run_schema

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"


@dataclass
class ProfilerResult:
    """Structured response when profiling completes."""

    run_dict: Dict[str, Any]
    output_path: Path


def profile_model(
    backend: str,
    model_spec: Dict[str, Any],
    config: Dict[str, Any],
    *,
    output_dir: Optional[Path | str] = None,
) -> ProfilerResult:
    """Execute ``backend`` adapter, validate, and persist the artifact."""

    adapter = get_adapter(backend)
    run_payload = adapter.run(model_spec, config)

    validate_run_schema(run_payload)

    target_dir = Path(output_dir) if output_dir else RUNS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = target_dir / f"run_{backend}_{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(run_payload, fh, indent=2, sort_keys=False)

    return ProfilerResult(run_dict=run_payload, output_path=output_path)


__all__ = ["ProfilerResult", "profile_model", "SchemaValidationError"]

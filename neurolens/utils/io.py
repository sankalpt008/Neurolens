"""JSON IO helpers for NeuroLens artifacts."""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict

__all__ = ["read_json", "write_json", "sha256_json"]


def read_json(path: str | Path) -> Dict[str, Any]:
    """Load a JSON file and return the decoded object."""

    file_path = Path(path)
    try:
        content = file_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:  # pragma: no cover - handled by caller in tests/cli
        raise
    except OSError as exc:
        raise RuntimeError(f"Failed to read JSON file '{file_path}': {exc}") from exc

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in '{file_path}': {exc}") from exc


def write_json(path: str | Path, obj: Dict[str, Any]) -> None:
    """Persist ``obj`` to ``path`` with deterministic formatting."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
    file_path.write_text(serialized + "\n", encoding="utf-8")


def sha256_json(obj: Any) -> str:
    """Return the SHA-256 hex digest of ``obj`` with canonical ordering."""

    canonical = _canonicalize(obj)
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


_VOLATILE_FIELDS = {"created_at", "updated_at", "timestamp"}


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key in sorted(value):
            if key in _VOLATILE_FIELDS:
                continue
            result[key] = _canonicalize(value[key])
        return result
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    return value

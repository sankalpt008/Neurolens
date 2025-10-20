"""Append-only run store with manifest and index helpers."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from neurolens.storage.manifest import append_manifest_line, iter_manifest
from neurolens.utils.io import ensure_dir, write_json
from neurolens.utils.validate import validate_run_schema

__all__ = [
    "write_run",
    "rebuild_index_from_manifest",
    "query_index",
    "ensure_dirs",
]

try:  # pragma: no cover - import probing
    import pandas as _pd  # type: ignore
    import pyarrow  # type: ignore  # noqa: F401 - imported for side effects
except Exception:  # pragma: no cover - pyarrow/pandas optional
    PARQUET_AVAILABLE = False
    _pd = None  # type: ignore
else:  # pragma: no cover - exercised when deps available
    PARQUET_AVAILABLE = True


def ensure_dirs(path: str | Path) -> Path:
    """Ensure ``path`` exists as a directory."""

    return ensure_dir(path)


def write_run(
    root: str | Path,
    run: Dict[str, Any],
    fingerprint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[str]]:
    """Persist ``run`` (and optional ``fingerprint``) into the store.

    The function validates the payload, writes JSON artefacts into a date/backend
    partition, appends a manifest entry, and updates the tabular index (Parquet when
    available, JSONL otherwise).
    """

    validate_run_schema(run)

    run_id = str(run.get("run_id") or "") or _generate_placeholder_run_id()
    created_at = _parse_datetime(run.get("created_at"))
    backend = str(run.get("software", {}).get("backend", "unknown"))
    precision = str(run.get("software", {}).get("precision", "unknown"))
    model_info = run.get("model", {}) if isinstance(run.get("model"), dict) else {}
    model_name = str(model_info.get("name", "unknown"))
    batch_size = _to_int(model_info.get("batch_size"))
    seq_len = _to_int(model_info.get("sequence_length"))
    if seq_len is None:
        seq_len = _to_int(model_info.get("seq_len"))
    total_latency = _to_float(run.get("summary", {}).get("total_duration_ms", 0.0))
    gpu_util = _to_float(run.get("summary", {}).get("metrics", {}).get("gpu_utilization", 0.0))

    meta = run.get("meta") if isinstance(run.get("meta"), dict) else {}
    env = meta.get("env", {}) if isinstance(meta.get("env"), dict) else {}
    meta_config = meta.get("config", {}) if isinstance(meta.get("config"), dict) else {}
    tags = meta.get("tags", {}) if isinstance(meta.get("tags"), dict) else {}

    if batch_size is None:
        batch_size = _to_int(meta_config.get("batch_size"))
    if seq_len is None:
        seq_len = _to_int(meta_config.get("seq_len"))
    if seq_len is None:
        seq_len = -1
    if batch_size is None:
        batch_size = -1

    driver_version = str(env.get("driver_version", "unknown"))
    cuda_version = str(env.get("cuda_version", "unknown"))
    gpu_name = str(env.get("gpu_name", "unknown"))

    root_path = ensure_dir(root)
    partition = _build_partition(root_path, created_at, backend)
    ensure_dir(partition)

    run_filename = f"{run_id}.json"
    run_path = partition / run_filename
    write_json(run_path, run)

    fp_rel_path: Optional[str] = None
    if fingerprint is not None:
        fp_filename = f"{run_id}.fp.json"
        fp_path = partition / fp_filename
        write_json(fp_path, fingerprint)
        fp_rel_path = _relative_path(fp_path, root_path)

    manifest_row = {
        "run_id": run_id,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "backend": backend,
        "model_name": model_name,
        "precision": precision,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "total_latency_ms": total_latency,
        "gpu_utilization": gpu_util,
        "gpu_name": gpu_name,
        "driver_version": driver_version,
        "cuda_version": cuda_version,
        "tags": json.dumps(tags, sort_keys=True),
        "path_run_json": _relative_path(run_path, root_path),
        "path_fp_json": fp_rel_path,
        "day": created_at.date().isoformat(),
    }

    append_manifest_line(root_path, manifest_row)
    _update_index(root_path, manifest_row)

    return {
        "run_json": manifest_row["path_run_json"],
        "fp_json": fp_rel_path,
    }


def rebuild_index_from_manifest(root: str | Path) -> str:
    """Reconstruct the tabular index from the manifest file."""

    root_path = ensure_dir(root)
    rows = [row for row in iter_manifest(root_path)]
    if not rows:
        index_path = root_path / ("_index.parquet" if PARQUET_AVAILABLE else "_index.jsonl")
        if index_path.exists():
            index_path.unlink()
        return str(index_path)

    return _write_index(root_path, rows)


def query_index(root: str | Path, limit: Optional[int] = None, **filters: Any) -> List[Dict[str, Any]]:
    """Return rows from the index filtered by ``filters``.

    Recognised filters: model_name, backend, gpu_name, precision, day, batch_size,
    seq_len, tags (list of ``key=value`` strings).
    """

    root_path = Path(root)
    rows = _load_index_rows(root_path)
    if not rows:
        return []

    tag_filters = filters.pop("tags", None)
    if tag_filters:
        tag_pairs = [_split_tag(tag_filter) for tag_filter in tag_filters]
    else:
        tag_pairs = []

    def _matches(row: Dict[str, Any]) -> bool:
        for key, value in filters.items():
            if value is None:
                continue
            if key not in row:
                return False
            current = row[key]
            if key in {"batch_size", "seq_len"}:
                try:
                    current = int(current)
                except (TypeError, ValueError):
                    return False
                if int(value) != current:
                    return False
            else:
                if str(current) != str(value):
                    return False
        if tag_pairs:
            row_tags = _parse_tags(row.get("tags"))
            for key, value in tag_pairs:
                if row_tags.get(key) != value:
                    return False
        return True

    matched = [row for row in rows if _matches(row)]
    if limit is not None:
        matched = matched[: int(limit)]
    return matched


# ---------------------------------------------------------------------------
# Internal helpers


def _generate_placeholder_run_id() -> str:
    return os.urandom(8).hex()


def _parse_datetime(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value.astimezone(dt.timezone.utc)
    if isinstance(value, str):
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            parsed = dt.datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.astimezone(dt.timezone.utc)
        except ValueError:
            pass
    return dt.datetime.now(dt.timezone.utc)


def _to_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float:
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _build_partition(root: Path, timestamp: dt.datetime, backend: str) -> Path:
    return root / f"{timestamp.year:04d}" / f"{timestamp.month:02d}" / f"{timestamp.day:02d}" / backend


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _update_index(root: Path, row: Dict[str, Any]) -> None:
    _write_index(root, [row], append=True)


def _write_index(root: Path, rows: List[Dict[str, Any]], append: bool = False) -> str:
    if PARQUET_AVAILABLE:
        return _write_parquet_index(root, rows, append=append)
    return _write_jsonl_index(root, rows, append=append)


def _write_parquet_index(root: Path, rows: List[Dict[str, Any]], append: bool = False) -> str:
    assert _pd is not None  # pragma: no cover - guarded by PARQUET_AVAILABLE
    index_path = root / "_index.parquet"
    df_new = _pd.DataFrame(rows)
    if append and index_path.exists():
        df_existing = _pd.read_parquet(index_path)
        df_combined = _pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_parquet(index_path, index=False)
    return str(index_path)


def _write_jsonl_index(root: Path, rows: List[Dict[str, Any]], append: bool = False) -> str:
    index_path = root / "_index.jsonl"
    mode = "a" if append and index_path.exists() else "w"
    with index_path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return str(index_path)


def _load_index_rows(root: Path) -> List[Dict[str, Any]]:
    parquet_path = root / "_index.parquet"
    jsonl_path = root / "_index.jsonl"

    if PARQUET_AVAILABLE and parquet_path.exists():
        df = _pd.read_parquet(parquet_path)  # type: ignore[operator]
        return [dict(record) for record in df.to_dict(orient="records")]

    if jsonl_path.exists():
        rows: List[Dict[str, Any]] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    # Fallback: rebuild from manifest if possible
    manifest_rows = [row for row in iter_manifest(root)]
    if not manifest_rows:
        return []
    _write_index(root, manifest_rows, append=False)
    return manifest_rows


def _split_tag(tag: str) -> tuple[str, str]:
    if "=" not in tag:
        return tag, ""
    key, value = tag.split("=", 1)
    return key, value


def _parse_tags(raw: Any) -> Dict[str, str]:
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    elif isinstance(raw, dict):
        data = raw
    else:
        return {}
    return {str(k): str(v) for k, v in data.items()}


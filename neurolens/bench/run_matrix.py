"""Grid runner for NeuroLens bench sweeps."""
from __future__ import annotations

import copy
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
from uuid import uuid4

from neurolens.bench.env import collect_env
from neurolens.core.profiler import profile_model
from neurolens.fingerprint.builder import build_fingerprint
from neurolens.utils.io import ensure_dir, write_json
from neurolens.utils.validate import validate_run_schema

__all__ = ["cfg_hash", "run_matrix"]


def cfg_hash(cfg: Mapping[str, Any]) -> str:
    """Return a short SHA1 hash for ``cfg`` to keep artifact names stable."""

    payload = json.dumps(dict(cfg), sort_keys=True)
    return _sha1(payload)[:8]


def run_matrix(
    model: str | Mapping[str, Any],
    backend: str,
    grid_cfg: Mapping[str, Any],
    out_dir: str | os.PathLike[str],
) -> List[Dict[str, Any]]:
    """Execute a sweep over ``grid_cfg`` combinations and persist artifacts."""

    runs_dir = ensure_dir(Path(out_dir) / "runs")
    fp_dir = ensure_dir(Path(out_dir) / "fp")

    env = collect_env()
    model_spec = _build_model_spec(backend, model)

    bs_list = _ensure_iterable(grid_cfg.get("batch_size", [1]))
    seq_list = _ensure_iterable(grid_cfg.get("seq_len", [128]))
    prec_list = _ensure_iterable(grid_cfg.get("precision", ["fp32"]))
    repeats = int(grid_cfg.get("repeats", 1))
    tags = grid_cfg.get("tags", {}) or {}
    allow_stub_trt = bool(grid_cfg.get("allow_stub_trt", False))

    rows: List[Dict[str, Any]] = []
    for bs, seq, prec in itertools.product(bs_list, seq_list, prec_list):
        for iteration in range(repeats):
            cfg = {
                "batch_size": int(bs),
                "sequence_length": int(seq),
                "precision": str(prec),
            }
            if allow_stub_trt:
                cfg["allow_stub_trt"] = True
            cfg["backend"] = backend

            run_identifier = _build_run_identifier(cfg)

            result = profile_model(
                backend,
                model_spec,
                cfg,
                output_dir=runs_dir,
            )
            run_dict = copy.deepcopy(result.run_dict)

            _augment_run_metadata(run_dict, env, tags, cfg, iteration)
            validate_run_schema(run_dict)

            run_path = runs_dir / f"{run_identifier}.json"
            write_json(run_path, run_dict)

            try:
                result.output_path.unlink()
            except FileNotFoundError:
                pass

            fingerprint = build_fingerprint(run_dict, peaks=None)
            fp_path = fp_dir / f"{run_identifier}.fp.json"
            write_json(fp_path, fingerprint)

            summary = run_dict.get("summary", {})
            total_latency = summary.get("total_duration_ms")
            if total_latency is None:
                total_latency = summary.get("total_latency_ms", 0.0)

            rows.append(
                {
                    "run_id": run_identifier,
                    "backend": backend,
                    "batch_size": int(bs),
                    "seq_len": int(seq),
                    "precision": str(prec),
                    "total_latency_ms": float(total_latency or 0.0),
                    "gpu_utilization": float(summary.get("gpu_utilization", 0.0)),
                    "model": run_dict.get("model", {}).get("name", "unknown"),
                    "gpu_name": env.get("gpu_name", "unknown"),
                }
            )

    _write_summary_csv(rows, Path(out_dir) / "summary.csv")
    return rows


def _ensure_iterable(value: Any) -> Iterable[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _build_model_spec(backend: str, model: str | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(model, Mapping):
        return dict(model)
    if isinstance(model, str):
        if model.startswith("builtin:"):
            return {"builtin": model.split(":", 1)[1]}
        return {"path": model}
    raise TypeError("Unsupported model specification")


def _build_run_identifier(cfg: Mapping[str, Any]) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}-{cfg_hash(cfg)}-{uuid4().hex[:6]}"


def _augment_run_metadata(
    run_dict: Dict[str, Any],
    env: Mapping[str, Any],
    tags: Mapping[str, Any],
    cfg: Mapping[str, Any],
    repeat_index: int,
) -> None:
    run_dict.setdefault("meta", {})
    meta = run_dict["meta"]
    if not isinstance(meta, dict):
        meta = {}
        run_dict["meta"] = meta
    meta["env"] = dict(env)
    meta["tags"] = {str(k): str(v) for k, v in dict(tags).items()}
    meta["config"] = {
        "batch_size": int(cfg.get("batch_size", 0)),
        "sequence_length": int(cfg.get("sequence_length", 0)),
        "precision": str(cfg.get("precision", "")),
        "backend": str(cfg.get("backend", "")),
        "repeat": int(repeat_index),
    }

    model = run_dict.get("model")
    if isinstance(model, dict):
        model["batch_size"] = int(cfg.get("batch_size", model.get("batch_size", 1)))


def _write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        import csv

        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha1(payload: str) -> str:
    from hashlib import sha1

    return sha1(payload.encode("utf-8")).hexdigest()

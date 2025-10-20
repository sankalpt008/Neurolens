"""Artifact bundling utilities for NeuroLens exports."""
from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import hashlib
import json
import os
from pathlib import Path
import zipfile
from typing import List

from neurolens.utils.io import ensure_dir, read_json

__all__ = ["bundle_artifacts"]


@dataclass
class _Artifact:
    path: Path
    arcname: str


def bundle_artifacts(run_path: str, fp_path: str, report_paths: List[str], out_dir: str) -> str:
    """Bundle profiling artifacts into a manifest-backed ZIP archive."""

    run_file = _require_file(run_path)
    fp_file = _require_file(fp_path)
    report_files = [_require_file(p) for p in report_paths]

    run_data = read_json(run_file)
    run_id = str(run_data.get("run_id")) if "run_id" in run_data else None

    timestamp = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    prefix = (run_id or timestamp).replace(os.sep, "_")
    bundle_dir = ensure_dir(out_dir)
    bundle_path = bundle_dir / f"{prefix}.bundle.zip"

    artifacts: List[_Artifact] = [
        _Artifact(run_file, f"runs/{run_file.name}"),
        _Artifact(fp_file, f"fingerprints/{fp_file.name}"),
    ]
    artifacts.extend(
        _Artifact(path, f"reports/{path.name}") for path in report_files
    )

    manifest_entries = [
        {
            "path": artifact.arcname,
            "sha256": _get_sha256(artifact.path),
            "size": artifact.path.stat().st_size,
        }
        for artifact in artifacts
    ]

    manifest = {
        "bundled_at": timestamp,
        "files": manifest_entries,
    }

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for artifact in artifacts:
            _safe_copy(artifact.path, artifact.arcname, zf)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    return str(bundle_path)


def _require_file(path: str) -> Path:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Required artifact not found: {file_path}")
    return file_path


def _get_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_copy(file_path: Path, arcname: str, zf: zipfile.ZipFile) -> None:
    zf.write(file_path, arcname)

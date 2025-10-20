from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

from typer.testing import CliRunner

from neurolens.cli import export as export_cli
from neurolens.exporter.bundle import bundle_artifacts
from neurolens.utils.io import write_json


def _make_sample_run() -> dict:
    return {
        "run_id": "test-run",
        "created_at": "2025-10-20T00:00:00Z",
        "hardware": {},
        "software": {},
        "model": {},
        "timeline": [],
        "summary": {},
    }


def _make_sample_fingerprint() -> dict:
    return {
        "run_id": "test-run",
        "source_run_sha": "deadbeef",
        "vector_spec": ["lat_norm"],
        "ops": [],
        "summary": {"total_latency_ms": 0.0, "num_ops": 0},
        "created_at": "2025-10-20T00:00:00Z",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_bundle_artifacts_creates_manifest(tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    fp_path = tmp_path / "run.fp.json"
    report_path = tmp_path / "report.html"

    write_json(run_path, _make_sample_run())
    write_json(fp_path, _make_sample_fingerprint())
    report_path.write_text("<html><body>report</body></html>\n", encoding="utf-8")

    out_dir = tmp_path / "exports"
    archive_path = Path(bundle_artifacts(str(run_path), str(fp_path), [str(report_path)], str(out_dir)))

    assert archive_path.exists()

    with zipfile.ZipFile(archive_path, "r") as zf:
        names = set(zf.namelist())
        assert names == {"runs/run.json", "fingerprints/run.fp.json", "reports/report.html", "manifest.json"}

        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["bundled_at"].endswith("Z")
        manifest_files = {entry["path"]: entry for entry in manifest["files"]}

        assert manifest_files["runs/run.json"]["sha256"] == _sha256(run_path)
        assert manifest_files["runs/run.json"]["size"] == run_path.stat().st_size
        assert manifest_files["fingerprints/run.fp.json"]["sha256"] == _sha256(fp_path)
        assert manifest_files["reports/report.html"]["sha256"] == _sha256(report_path)


def test_export_cli_creates_bundle(tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    fp_path = tmp_path / "run.fp.json"
    report_path = tmp_path / "report.html"
    write_json(run_path, _make_sample_run())
    write_json(fp_path, _make_sample_fingerprint())
    report_path.write_text("<html>report</html>\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        export_cli.app,
        [
            "--run",
            str(run_path),
            "--fingerprint",
            str(fp_path),
            "--report",
            str(report_path),
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 0
    assert "Bundle created at" in result.stdout
    bundle_line = [line for line in result.stdout.splitlines() if line.startswith("Bundle created at")][0]
    bundle_path = Path(bundle_line.split(" ", 3)[-1])
    assert bundle_path.exists()

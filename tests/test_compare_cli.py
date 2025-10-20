from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from neurolens.cli import compare as compare_cli
from neurolens.fingerprint import build_fingerprint
from neurolens.utils.io import read_json, write_json


def test_compare_cli_generates_report(tmp_path: Path) -> None:
    runner = CliRunner()
    run_dict = read_json(Path("samples/trace_minimal.json"))
    baseline_fp = build_fingerprint(run_dict)

    modified_run = dict(run_dict)
    modified_timeline = []
    for entry in run_dict["timeline"]:
        copy_entry = dict(entry)
        copy_entry["duration_ms"] = entry["duration_ms"] * 1.1
        copy_entry["metrics"] = dict(entry.get("metrics", {}))
        copy_entry["metrics"]["latency_ms"] = copy_entry["duration_ms"]
        modified_timeline.append(copy_entry)
    modified_run["timeline"] = modified_timeline
    modified_run["summary"] = dict(run_dict["summary"])
    modified_run["summary"]["total_duration_ms"] = sum(e["duration_ms"] for e in modified_timeline)

    candidate_fp = build_fingerprint(modified_run)

    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    markdown_path = tmp_path / "report.md"

    write_json(baseline_path, baseline_fp)
    write_json(candidate_path, candidate_fp)

    result = runner.invoke(
        compare_cli.app,
        [
            "--a",
            str(baseline_path),
            "--b",
            str(candidate_path),
            "--topk",
            "2",
            "--markdown",
            str(markdown_path),
        ],
    )

    assert result.exit_code == 0
    assert "Similarity:" in result.stdout
    assert markdown_path.exists()
    assert "Δ latency" in markdown_path.read_text(encoding="utf-8")

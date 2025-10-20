from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from neurolens.cli import report as report_cli
from neurolens.utils.io import write_json

from helpers import build_insights_sample_run


def test_report_cli_writes_markdown(tmp_path: Path) -> None:
    run_path = tmp_path / "run.json"
    write_json(run_path, build_insights_sample_run())

    output_path = tmp_path / "report.md"
    runner = CliRunner()
    result = runner.invoke(report_cli.app, ["--run", str(run_path), "--md", str(output_path)])

    assert result.exit_code == 0
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "NeuroLens Profiling Report" in content
    assert "memory_bound_low_ai" in content

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from neurolens.cli import view as view_cli


def test_view_cli_embedded(monkeypatch, tmp_path: Path) -> None:
    runner = CliRunner()
    run_file = tmp_path / "run.json"
    run_file.write_text(Path("samples/trace_minimal.json").read_text(encoding="utf-8"), encoding="utf-8")

    called = {}

    def fake_run_app(preload) -> None:  # type: ignore[no-untyped-def]
        called["preload"] = preload

    monkeypatch.setenv("NEUROLENS_VIEW_EMBEDDED", "1")
    monkeypatch.setattr(view_cli, "run_app", fake_run_app)

    result = runner.invoke(view_cli.app, ["--run", str(run_file)])

    assert result.exit_code == 0
    assert "preload" in called
    assert called["preload"].run_path == run_file

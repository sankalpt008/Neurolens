"""CLI for bundling NeuroLens artifacts."""
from __future__ import annotations

from typing import List

import typer

from neurolens.exporter.bundle import bundle_artifacts

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    help="Bundle runs, fingerprints, and reports into shareable archives.",
)


@app.callback()
def export(
    run: str = typer.Option(..., help="Path to run.json"),
    fingerprint: str = typer.Option(..., help="Path to fingerprint .fp.json"),
    report: List[str] = typer.Option([], help="Report files (HTML/MD)"),
    out_dir: str = typer.Option("exports", help="Output directory for the bundle"),
) -> None:
    """Create a NeuroLens bundle archive."""

    try:
        out_path = bundle_artifacts(run, fingerprint, report, out_dir)
    except Exception as exc:  # pragma: no cover - propagate friendly error
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    typer.echo(f"Bundle created at {out_path}")


__all__ = ["app", "export"]

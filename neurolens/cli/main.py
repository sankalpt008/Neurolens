"""Root CLI entry point for NeuroLens."""
from __future__ import annotations

import typer

from . import bench as bench_cli
from . import compare as compare_cli
from . import export as export_cli
from . import fingerprint as fingerprint_cli
from . import ingest as ingest_cli
from . import ls as ls_cli
from . import profile as profile_cli
from . import report as report_cli
from . import view as view_cli

app = typer.Typer(add_completion=False, help="NeuroLens command line interface")
app.add_typer(profile_cli.app, name="profile", help="Profile a model with a chosen backend")
app.add_typer(ingest_cli.app, name="ingest", help="Ingest existing run artifacts into storage")
app.add_typer(fingerprint_cli.app, name="fingerprint", help="Build fingerprints from runs")
app.add_typer(compare_cli.app, name="compare", help="Compare two fingerprints")
app.add_typer(report_cli.app, name="report", help="Generate Markdown/HTML insight reports")
app.add_typer(ls_cli.app, name="ls", help="List stored runs with optional filters")
app.add_typer(bench_cli.app, name="bench", help="Sweep batch/seq/precision grids")
app.add_typer(export_cli.app, name="export", help="Bundle runs, fingerprints, and reports")
app.add_typer(view_cli.app, name="view", help="Launch the local visualization dashboard")


def run() -> None:
    """Execute the root CLI."""

    app()


__all__ = ["app", "run"]

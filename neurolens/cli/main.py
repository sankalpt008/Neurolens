"""Root CLI entry point for NeuroLens."""
from __future__ import annotations

import typer

from . import bench as bench_cli
from . import compare as compare_cli
from . import fingerprint as fingerprint_cli
from . import profile as profile_cli
from . import report as report_cli
from . import export as export_cli
from . import view as view_cli

app = typer.Typer(add_completion=False, help="NeuroLens command line interface")
app.add_typer(profile_cli.app, name="profile")
app.add_typer(fingerprint_cli.app, name="fingerprint")
app.add_typer(compare_cli.app, name="compare")
app.add_typer(report_cli.app, name="report")
app.add_typer(view_cli.app, name="view")
app.add_typer(export_cli.app, name="export")
app.add_typer(bench_cli.app, name="bench")


def run() -> None:
    """Execute the root CLI."""

    app()


__all__ = ["app", "run"]

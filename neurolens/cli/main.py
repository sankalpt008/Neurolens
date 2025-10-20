"""Root CLI entry point for NeuroLens."""
from __future__ import annotations

import typer

from . import compare as compare_cli
from . import fingerprint as fingerprint_cli
from . import profile as profile_cli
from . import report as report_cli

app = typer.Typer(add_completion=False, help="NeuroLens command line interface")
app.add_typer(profile_cli.app, name="profile")
app.add_typer(fingerprint_cli.app, name="fingerprint")
app.add_typer(compare_cli.app, name="compare")
app.add_typer(report_cli.app, name="report")


def run() -> None:
    """Execute the root CLI."""

    app()


__all__ = ["app", "run"]

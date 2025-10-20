"""CLI helpers to ingest existing profiling runs into the NeuroLens store."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from neurolens.fingerprint.builder import build_fingerprint
from neurolens.storage.store import write_run
from neurolens.utils.io import read_json
from neurolens.utils.validate import SchemaValidationError, validate_run_schema

app = typer.Typer(
    add_completion=False,
    help="Ingest previously generated run artifacts into the NeuroLens storage layer.",
    invoke_without_command=True,
)


@app.callback()
def ingest(  # type: ignore[override]
    run: Path = typer.Option(
        ..., "--run", path_type=Path, exists=True, readable=True, help="Path to a run.json artifact"
    ),
    fingerprint: Optional[Path] = typer.Option(
        None,
        "--fingerprint",
        path_type=Path,
        exists=True,
        readable=True,
        help="Optional fingerprint JSON to import alongside the run",
    ),
    auto_fingerprint: bool = typer.Option(
        True,
        "--auto-fingerprint/--no-auto-fingerprint",
        help="Automatically build a fingerprint when one is not provided.",
    ),
    root: Path = typer.Option(
        Path("runs_store"),
        "--root",
        help="Destination run-store root directory (defaults to ./runs_store)",
    ),
) -> None:
    """Validate and ingest run (and optional fingerprint) artifacts."""

    try:
        run_dict = read_json(run)
    except Exception as exc:  # pragma: no cover - error formatting only
        typer.secho(f"[ERROR] Failed to read run JSON: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    try:
        validate_run_schema(run_dict)
    except SchemaValidationError as exc:
        typer.secho("[ERROR] Run failed schema validation:", fg=typer.colors.RED)
        typer.echo(exc.message)
        raise typer.Exit(code=1) from exc

    fingerprint_dict = None
    if fingerprint is not None:
        try:
            fingerprint_dict = read_json(fingerprint)
        except Exception as exc:  # pragma: no cover - defensive formatting
            typer.secho(f"[ERROR] Failed to read fingerprint JSON: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1) from exc
    elif auto_fingerprint:
        fingerprint_dict = build_fingerprint(run_dict)

    try:
        paths = write_run(root, run_dict, fingerprint=fingerprint_dict)
    except Exception as exc:  # pragma: no cover - storage writes are heavily unit-tested
        typer.secho(f"[ERROR] Failed to write artifacts to store: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.secho(
        "Ingestion complete:",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"  run_json: {paths.get('run_json')}")
    if paths.get("fp_json"):
        typer.echo(f"  fp_json: {paths['fp_json']}")


__all__ = ["app"]

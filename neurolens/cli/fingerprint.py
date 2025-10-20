"""CLI for building NeuroLens fingerprints."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import typer

from neurolens.fingerprint.builder import build_fingerprint
from neurolens.utils.io import read_json, write_json
from neurolens.utils.validate import SchemaValidationError, validate_run_schema

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    help="Build normalized fingerprints from run artifacts.",
)


def _parse_peaks(peaks: str | None) -> Dict[str, float | None]:
    if not peaks:
        return {}
    result: Dict[str, float | None] = {}
    for part in peaks.split(","):
        if not part:
            continue
        if "=" not in part:
            raise typer.BadParameter("Peak overrides must use key=value syntax")
        key, raw_value = part.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key not in {"peak_dram_gbps", "sm_count"}:
            raise typer.BadParameter(f"Unsupported peak key '{key}'")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise typer.BadParameter(f"Invalid numeric value for '{key}': {raw_value}") from exc
        result[key] = value
    return result


@app.callback()
def fingerprint(
    run: Path = typer.Option(..., "--run", exists=True, readable=True, path_type=Path, help="Input run JSON"),
    out: Path = typer.Option(..., "--out", path_type=Path, help="Destination fingerprint JSON"),
    peaks: str | None = typer.Option(None, "--peaks", help="Comma-delimited hardware peaks e.g. peak_dram_gbps=1555"),
) -> None:
    """Build a fingerprint and persist it to ``out``."""

    try:
        run_dict = read_json(run)
    except Exception as exc:
        typer.secho(f"[ERROR] Failed to load run JSON: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    try:
        validate_run_schema(run_dict)
    except SchemaValidationError as exc:
        typer.secho("[ERROR] Input run failed schema validation:", fg=typer.colors.RED)
        typer.echo(exc.message)
        raise typer.Exit(code=1) from exc

    peak_overrides = _parse_peaks(peaks)
    fingerprint_dict = build_fingerprint(run_dict, peak_overrides)

    try:
        write_json(out, fingerprint_dict)
    except Exception as exc:
        typer.secho(f"[ERROR] Failed to write fingerprint: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    typer.secho(
        f"Fingerprint written to {out} (ops={len(fingerprint_dict['ops'])}, sha={fingerprint_dict['source_run_sha'][:8]})",
        fg=typer.colors.GREEN,
    )


def run() -> None:
    """Entrypoint for ``python -m neurolens.cli.fingerprint`` usage."""

    app()


__all__ = ["app", "fingerprint", "run"]

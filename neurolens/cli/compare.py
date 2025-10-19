"""CLI for comparing NeuroLens fingerprints."""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import typer

from neurolens.fingerprint.similarity import diff
from neurolens.utils.io import read_json

app = typer.Typer(
    add_completion=False,
    invoke_without_command=True,
    help="Compare two fingerprints and highlight divergences.",
)


def _load_fingerprint(path: Path) -> dict:
    try:
        return read_json(path)
    except Exception as exc:
        typer.secho(f"[ERROR] Failed to load fingerprint '{path}': {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc


def _format_table(rows: List[Tuple[str, str, float | None]]) -> str:
    header = "Op | Type | Δ latency %"
    separator = "---|---|---"
    body = "\n".join(
        f"{name} | {op_type} | {delta if delta is not None else 'n/a'}"
        for name, op_type, delta in rows
    )
    return f"{header}\n{separator}\n{body}" if body else f"{header}\n{separator}\n"


@app.callback()
def compare(
    a: Path = typer.Option(..., "--a", exists=True, readable=True, path_type=Path, help="Baseline fingerprint"),
    b: Path = typer.Option(..., "--b", exists=True, readable=True, path_type=Path, help="Target fingerprint"),
    topk: int = typer.Option(5, "--topk", min=1, help="Number of divergent ops to display"),
    markdown: Path | None = typer.Option(
        None,
        "--markdown",
        path_type=Path,
        help="Optional Markdown output path",
    ),
) -> None:
    """Compare fingerprints ``a`` and ``b``."""

    fp_a = _load_fingerprint(a)
    fp_b = _load_fingerprint(b)

    report = diff(fp_a, fp_b, topk=topk)

    typer.echo(f"Similarity: {report['similarity']:.4f}")
    typer.echo(f"Unmatched ops → baseline: {report['unmatched_a']}, candidate: {report['unmatched_b']}")

    rows: List[Tuple[str, str, float | None]] = []
    for entry in report["top_divergences"]:
        rows.append(
            (
                str(entry.get("name")),
                str(entry.get("type")),
                entry.get("latency_pct_change"),
            )
        )
    if rows:
        typer.echo("Top divergences:")
        for row in rows:
            typer.echo(f"  - {row[0]} ({row[1]}): Δlat%={row[2] if row[2] is not None else 'n/a'}")

    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(_format_table(rows) + "\n", encoding="utf-8")
        typer.echo(f"Markdown report written to {markdown}")


def run() -> None:
    """Entrypoint for ``python -m neurolens.cli.compare`` usage."""

    app()


__all__ = ["app", "compare", "run"]

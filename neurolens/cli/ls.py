"""CLI for listing stored NeuroLens runs."""
from __future__ import annotations

from typing import List, Optional

import json
import os

import typer

from neurolens.storage.store import query_index

app = typer.Typer(help="List stored runs from the NeuroLens manifest/index", add_completion=False, invoke_without_command=True)


@app.callback()
def ls(  # type: ignore[override]
    root: Optional[str] = typer.Option(None, help="Root directory of the run store"),
    model: Optional[str] = typer.Option(None, help="Filter by model name"),
    backend: Optional[str] = typer.Option(None, help="Filter by backend"),
    gpu: Optional[str] = typer.Option(None, help="Filter by GPU name"),
    precision: Optional[str] = typer.Option(None, help="Filter by precision"),
    day: Optional[str] = typer.Option(None, help="Filter by UTC day (YYYY-MM-DD)"),
    batch: Optional[int] = typer.Option(None, help="Filter by batch size"),
    seq: Optional[int] = typer.Option(None, help="Filter by sequence length"),
    tag: List[str] = typer.Option([], help="Filter by tag key=value"),
    limit: Optional[int] = typer.Option(50, help="Maximum rows to display (None for all)"),
    json_output: bool = typer.Option(False, "--json", help="Emit JSON instead of a table"),
) -> None:
    """List stored runs applying optional filters."""

    store_root = root or os.getenv("NEUROLENS_STORE_ROOT", "runs_store")

    filters = {
        "model_name": model,
        "backend": backend,
        "gpu_name": gpu,
        "precision": precision,
        "day": day,
        "batch_size": batch,
        "seq_len": seq,
        "tags": tag,
    }

    rows = query_index(store_root, limit=limit, **filters)
    if json_output:
        typer.echo(json.dumps(rows, ensure_ascii=False, indent=2))
        return

    if not rows:
        typer.echo("No runs matched the provided filters.")
        return

    header = ["run_id", "day", "backend", "model", "bs", "seq", "prec", "total_ms", "gpu"]
    lines = [" | ".join(header)]
    lines.append("-" * len(lines[0]))
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row.get("run_id", "")),
                    str(row.get("day", "")),
                    str(row.get("backend", "")),
                    str(row.get("model_name", "")),
                    str(row.get("batch_size", "")),
                    str(row.get("seq_len", "")),
                    str(row.get("precision", "")),
                    str(row.get("total_latency_ms", "")),
                    str(row.get("gpu_name", "")),
                ]
            )
        )
    typer.echo("\n".join(lines))


__all__ = ["app"]


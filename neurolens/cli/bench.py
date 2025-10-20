"""CLI entrypoint for NeuroLens bench harness."""
from __future__ import annotations

from pathlib import Path

import typer

try:  # pragma: no cover - dependency may be missing in minimal envs
    import yaml
except ImportError:  # pragma: no cover - handled at runtime
    yaml = None

from neurolens.bench.run_matrix import run_matrix

app = typer.Typer(add_completion=False, help="Run batch-size/sequence-length sweeps.")


@app.command()
def run(
    config: str = typer.Option(..., "--config", help="Path to bench YAML configuration."),
    out_dir: str = typer.Option(
        "bench_runs", "--out-dir", help="Directory to store run artifacts and summaries."
    ),
) -> None:
    """Execute the bench harness using ``config``."""

    config_path = Path(config)
    if not config_path.exists():
        typer.secho(f"[ERROR] Bench config not found: {config_path}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    if yaml is None:
        typer.secho("[ERROR] PyYAML is required for the bench CLI. Install with 'pip install pyyaml'.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    backend = cfg.get("backend")
    model = cfg.get("model")
    if not backend or not model:
        typer.secho("[ERROR] Config must include 'backend' and 'model'.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    grid_cfg = dict(cfg.get("grid", {}))
    if "repeats" not in grid_cfg and "repeats" in cfg:
        grid_cfg["repeats"] = cfg.get("repeats")
    if "tags" not in grid_cfg and "tags" in cfg:
        grid_cfg["tags"] = cfg.get("tags")
    if "allow_stub_trt" not in grid_cfg and cfg.get("allow_stub_trt"):
        grid_cfg["allow_stub_trt"] = bool(cfg.get("allow_stub_trt"))

    rows = run_matrix(model, str(backend), grid_cfg, out_dir)
    typer.echo(
        f"Completed {len(rows)} runs. Summary CSV: {Path(out_dir) / 'summary.csv'}"
    )


def run_cli() -> None:  # pragma: no cover - compatibility shim
    app()


__all__ = ["app", "run", "run_cli"]

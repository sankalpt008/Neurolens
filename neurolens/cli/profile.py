# neurolens/cli/profile.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import typer

from neurolens.core.profiler import ProfilerResult, SchemaValidationError, profile_model

# We export only a command function; the Typer app lives in cli/main.py
BACKENDS = ("onnxrt", "torch", "tensorrt")


def _build_model_spec(backend: str, model: str) -> Dict[str, str]:
    """
    Convert user input into a model spec understood by adapters.
    - torch + 'tiny_linear' is a built-in toy
    - otherwise `model` must be a valid path
    """
    path = Path(model)
    if backend == "torch" and model == "tiny_linear":
        return {"builtin": "tiny_linear"}
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {model}")
    return {"path": str(path.resolve())}


def _print_summary(result: ProfilerResult) -> None:
    timeline = result.run_dict.get("timeline", []) or []
    summary = result.run_dict.get("summary", {}) or {}
    total_ops = len(timeline)
    total_latency = float(summary.get("total_duration_ms", 0.0))

    top_ops = sorted(timeline, key=lambda op: op.get("duration_ms", 0.0), reverse=True)[:3]
    top_ops_str = ", ".join(
        f"{entry.get('op_name','?')} ({entry.get('duration_ms',0.0):.3f} ms)" for entry in top_ops
    ) or "n/a"

    typer.secho(
        f"Ran {total_ops} ops, total {total_latency:.3f} ms, top-3: {top_ops_str}",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Artifact written to {result.output_path}")


# This function is registered as a Typer command by cli/main.py
def profile(
    backend: str = typer.Option("onnxrt", "--backend", help="Runtime backend", case_sensitive=False),
    model: str = typer.Option(..., "--model", help="Path or builtin model"),
    bs: int = typer.Option(1, "--bs", "--batch-size", min=1, help="Batch size override"),
    seq: int = typer.Option(1, "--seq", "--sequence-length", min=1, help="Sequence length override"),
    precision: str = typer.Option("fp32", "--precision", help="Target numerical precision"),
    allow_stub_trt: bool = typer.Option(
        False,
        "--allow-stub-trt",
        help="Permit emitting stub TensorRT artifacts when TensorRT is unavailable",
    ),
) -> None:
    """Profile a model with the selected backend and persist the artifact."""
    backend = backend.lower()
    if backend not in BACKENDS:
        typer.secho(f"[ERROR] Unsupported backend '{backend}'. Choose from {BACKENDS}.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Build model spec
    try:
        model_spec = _build_model_spec(backend, model)
    except FileNotFoundError as exc:
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    # Build run config
    config: Dict[str, Any] = {"batch_size": bs, "sequence_length": seq, "precision": precision}
    if allow_stub_trt:
        config["allow_stub_trt"] = True

    # Run profiling
    try:
        result = profile_model(backend, model_spec, config)
    except SchemaValidationError as exc:
        typer.secho("[ERROR] Generated artifact failed schema validation:", fg=typer.colors.RED)
        typer.echo(exc.message)
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # safety net
        typer.secho(f"[ERROR] Unexpected failure: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    _print_summary(result)


__all__ = ["profile"]


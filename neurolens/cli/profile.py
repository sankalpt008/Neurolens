"""Command line entrypoints for NeuroLens profiling."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import typer

from neurolens.core.profiler import ProfilerResult, SchemaValidationError, profile_model

app = typer.Typer(add_completion=False, help="Profile models with NeuroLens adapters.")

BACKENDS = ("onnxrt", "torch", "tensorrt")


def _build_model_spec(backend: str, model: str) -> Dict[str, str]:
    path = Path(model)
    if backend == "torch" and model == "tiny_linear":
        return {"builtin": "tiny_linear"}
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {model}")
    return {"path": str(path)}


@app.command()
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
        typer.secho(f"[ERROR] Unsupported backend '{backend}'.", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        model_spec = _build_model_spec(backend, model)
    except FileNotFoundError as exc:
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    config = {
        "batch_size": bs,
        "sequence_length": seq,
        "precision": precision,
    }
    if allow_stub_trt:
        config["allow_stub_trt"] = True

    try:
        result = profile_model(backend, model_spec, config)
    except SchemaValidationError as exc:
        typer.secho("[ERROR] Generated artifact failed schema validation:", fg=typer.colors.RED)
        typer.echo(exc.message)
        raise typer.Exit(code=1) from exc
    except RuntimeError as exc:
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # pragma: no cover - safety net for unexpected issues
        typer.secho(f"[ERROR] Unexpected failure: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    _print_summary(result)


def _print_summary(result: ProfilerResult) -> None:
    timeline = result.run_dict.get("timeline", [])
    summary = result.run_dict.get("summary", {})
from typing import Optional

import typer

from neurolens.adapters.onnxrt_adapter import OnnxRuntimeAdapter, OnnxRuntimeNotAvailable
from neurolens.core.profiler import Profiler, SchemaValidationError

app = typer.Typer(add_completion=False, help="Profile models with NeuroLens adapters.")


@app.command()
def profile(
    model: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True, help="Path to an ONNX model"),
    bs: Optional[int] = typer.Option(None, "--bs", "--batch-size", min=1, help="Override batch size for dynamic dims"),
    seq: Optional[int] = typer.Option(
        None, "--seq", "--sequence-length", min=1, help="Override sequence length for dynamic dims"
    ),
    precision: str = typer.Option("fp32", "--precision", help="Target numerical precision"),
) -> None:
    """Profile an ONNX model and persist a schema-compliant artifact."""

    adapter = OnnxRuntimeAdapter()
    profiler = Profiler(adapter)
    try:
        result = profiler.profile(model, batch_size=bs, sequence_length=seq, precision=precision)
    except OnnxRuntimeNotAvailable as exc:
        typer.secho(f"[ERROR] {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc
    except SchemaValidationError as exc:
        typer.secho("[ERROR] Generated artifact failed schema validation:", fg=typer.colors.RED)
        typer.echo(exc.errors)
        raise typer.Exit(code=1) from exc

    timeline = result.run_dict["timeline"]
    summary = result.run_dict["summary"]
    total_ops = len(timeline)
    total_latency = summary.get("total_duration_ms", 0.0)
    top_ops = sorted(timeline, key=lambda op: op.get("duration_ms", 0.0), reverse=True)[:3]
    top_ops_str = ", ".join(
        f"{entry['op_name']} ({entry['duration_ms']:.3f} ms)" for entry in top_ops
    ) or "n/a"
    )

    typer.secho(
        f"Ran {total_ops} ops, total {total_latency:.3f} ms, top-3: {top_ops_str}",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Artifact written to {result.output_path}")


def run() -> None:
    """Entrypoint for ``python -m neurolens.cli.profile`` usage."""

    app()


__all__ = ["app", "profile", "run"]

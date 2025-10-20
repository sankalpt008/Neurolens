"""CLI launcher for the NeuroLens visualization dashboard."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import typer

from neurolens.viz.dashboard import PreloadedArtifacts, run_app

app = typer.Typer(add_completion=False, help="Launch the local NeuroLens dashboard")


def _prepare_preload(
    run: Optional[Path],
    fingerprint: Optional[Path],
    compare: Optional[Tuple[Path, Path]],
) -> PreloadedArtifacts:
    baseline = compare[0] if compare else None
    candidate = compare[1] if compare else None
    return PreloadedArtifacts(
        run_path=run,
        fingerprint_path=fingerprint,
        baseline_path=baseline,
        compare_path=candidate,
    )


@app.command()
def start(
    run: Optional[Path] = typer.Option(None, "--run", help="Path to run.json to preload"),
    fingerprint: Optional[Path] = typer.Option(
        None, "--fingerprint", help="Path to fingerprint JSON to preload"
    ),
    compare: Optional[Tuple[Path, Path]] = typer.Option(
        None,
        "--compare",
        metavar="BASELINE CANDIDATE",
        help="Fingerprint pair for diff mode",
    ),
    port: int = typer.Option(8501, "--port", help="Port for Streamlit server"),
) -> None:
    """Start the local NeuroLens visualization dashboard."""

    preload = _prepare_preload(run, fingerprint, compare)

    if os.environ.get("NEUROLENS_VIEW_EMBEDDED") == "1":
        run_app(preload)
        return

    try:
        import streamlit  # noqa: F401  # pylint: disable=unused-import
    except ImportError as exc:  # pragma: no cover - runtime guard
        raise typer.Exit(code=1, message="Streamlit is required for neurolens view") from exc

    script = Path(__file__).resolve().parents[1] / "viz" / "dashboard.py"
    env = os.environ.copy()
    if preload.run_path:
        env["NEUROLENS_VIEW_RUN"] = str(preload.run_path)
    if preload.fingerprint_path:
        env["NEUROLENS_VIEW_FP"] = str(preload.fingerprint_path)
    if preload.baseline_path:
        env["NEUROLENS_VIEW_BASELINE"] = str(preload.baseline_path)
    if preload.compare_path:
        env["NEUROLENS_VIEW_CANDIDATE"] = str(preload.compare_path)

    typer.echo(f"Starting dashboard at http://localhost:{port}")

    try:
        subprocess.run(  # noqa: S603,S607 - intentional CLI passthrough
            [
                "streamlit",
                "run",
                str(script),
                "--server.port",
                str(port),
            ],
            env=env,
            check=True,
        )
    except KeyboardInterrupt:  # pragma: no cover - user interruption
        typer.echo("Dashboard stopped")
        raise typer.Exit(code=0)


__all__ = ["app", "start"]

"""Environment capture helpers for NeuroLens bench harness."""
from __future__ import annotations

import platform
import subprocess
from typing import Optional

__all__ = [
    "collect_env",
    "get_cuda_version",
    "get_driver_version",
    "get_gpu_name",
]


def _run(cmd: list[str]) -> Optional[str]:
    """Run ``cmd`` returning stripped stdout or ``None`` on failure."""

    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.STDOUT, text=True, timeout=5
        ).strip()
    except Exception:
        return None


def get_cuda_version() -> str:
    """Return CUDA version information if available."""

    nvcc = _run(["nvcc", "--version"])
    if nvcc and "release" in nvcc:
        return nvcc
    smi = _run(["nvidia-smi"])
    return smi or "unknown"


def get_driver_version() -> str:
    """Return the installed GPU driver version or ``unknown``."""

    smi = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if smi:
        first = smi.splitlines()[0].strip()
        return first or "unknown"
    return "unknown"


def get_gpu_name() -> str:
    """Return the primary GPU name when available."""

    smi = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if smi:
        first = smi.splitlines()[0].strip()
        return first or "unknown"
    return "unknown"


def collect_env() -> dict[str, str]:
    """Collect stable environment metadata for bench outputs."""

    return {
        "driver_version": get_driver_version() or "unknown",
        "cuda_version": get_cuda_version() or "unknown",
        "gpu_name": get_gpu_name() or "unknown",
        "os": f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
    }

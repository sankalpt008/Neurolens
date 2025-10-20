"""Environment detection utilities."""
from __future__ import annotations

import os
import platform
from typing import Any, Dict


def _detect_gpu_with_torch() -> Dict[str, Any] | None:
    try:
        import torch  # type: ignore
    except Exception:
        return None

    if not torch.cuda.is_available():
        return None

    index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(index)
    total_mem_gb = torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
    driver = torch.version.cuda or "unknown"
    return {
        "vendor": "NVIDIA",
        "gpu_model": name,
        "gpu_count": torch.cuda.device_count(),
        "memory_gb": round(total_mem_gb, 2),
        "driver_version": driver,
    }


def _detect_gpu_with_env() -> Dict[str, Any] | None:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible:
        return None
    devices = [dev for dev in cuda_visible.split(",") if dev.strip()]
    if not devices:
        return None
    return {
        "vendor": "NVIDIA",
        "gpu_model": "Unknown GPU",
        "gpu_count": len(devices),
        "memory_gb": 1.0,
        "driver_version": os.environ.get("CUDA_DRIVER_VERSION", "unknown"),
    }


def get_hardware_info() -> Dict[str, Any]:
    """Return a schema-compatible hardware description."""

    gpu_info = _detect_gpu_with_torch() or _detect_gpu_with_env()
    if gpu_info:
        return gpu_info

    # CPU fallback
    return {
        "vendor": "CPU",
        "gpu_model": platform.processor() or platform.machine() or "GenericCPU",
        "gpu_count": 1,
        "memory_gb": _detect_system_memory_gb(),
        "driver_version": platform.version(),
    }


def _detect_system_memory_gb() -> float:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = pages * page_size
        return round(total / (1024 ** 3), 2)
    except (ValueError, OSError, AttributeError):
        return 1.0


def get_software_info(backend: str, precision: str) -> Dict[str, Any]:
    """Return a schema-compatible software description."""

    version_lookup = {
        "onnxrt": "onnxruntime",
        "torch": "torch",
        "tensorrt": "tensorrt",
    }
    module_name = version_lookup.get(backend)
    version = "unknown"
    if module_name:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", version)
        except Exception:
            version = "unknown"
    return {
        "backend": backend,
        "version": version,
        "precision": precision,
    }

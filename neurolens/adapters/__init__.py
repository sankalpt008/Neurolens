"""Adapter implementations and registry exports."""
from .base import Adapter, REGISTRY, get_adapter, register
from .onnxrt_adapter import OnnxRuntimeAdapter, OnnxRuntimeNotAvailable
from .torch_adapter import TorchAdapter
from .tensorrt_adapter import TensorRTAdapter

__all__ = [
    "Adapter",
    "REGISTRY",
    "get_adapter",
    "register",
    "OnnxRuntimeAdapter",
    "OnnxRuntimeNotAvailable",
    "TorchAdapter",
    "TensorRTAdapter",
]
"""Backend adapters translating profiler outputs into the NeuroLens schema."""

from .onnxrt_adapter import OnnxRuntimeAdapter, OnnxRuntimeNotAvailable

__all__ = ["OnnxRuntimeAdapter", "OnnxRuntimeNotAvailable"]

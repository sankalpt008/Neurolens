"""Adapter protocol and registry."""
from __future__ import annotations

from typing import Any, Callable, Dict, Protocol

AdapterFactory = Callable[[], "Adapter"]


class Adapter(Protocol):
    """Runtime adapter returning schema-compliant profiling payloads."""

    name: str

    def run(self, model_spec: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute ``model_spec`` and return a schema-valid profiling dict."""


REGISTRY: Dict[str, AdapterFactory] = {}


def register(adapter_cls: Callable[[], Adapter]) -> Callable[[], Adapter]:
    """Class decorator registering an adapter implementation."""

    name = getattr(adapter_cls, "name", None)
    if not name:
        raise ValueError("Adapters must define a 'name' attribute for registration")
    REGISTRY[name] = adapter_cls  # type: ignore[assignment]
    return adapter_cls


def get_adapter(name: str) -> Adapter:
    """Return an instantiated adapter by ``name``."""

    if name not in REGISTRY:
        raise KeyError(f"Unknown adapter '{name}'. Registered: {', '.join(sorted(REGISTRY))}")
    return REGISTRY[name]()

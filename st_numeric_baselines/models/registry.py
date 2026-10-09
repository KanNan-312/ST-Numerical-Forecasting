from __future__ import annotations

from typing import Callable, Dict, Type

from .base import BaseForecaster


_REGISTRY: Dict[str, Callable[[], BaseForecaster]] = {}


def register(name: str) -> Callable[[Callable[[], BaseForecaster]], Callable[[], BaseForecaster]]:
    """Decorator for registering model factories."""
    key = name.lower()

    def _wrap(factory: Callable[[], BaseForecaster]) -> Callable[[], BaseForecaster]:
        if key in _REGISTRY:
            raise KeyError(f"Model '{name}' already registered")
        _REGISTRY[key] = factory
        return factory

    return _wrap


def get(name: str) -> BaseForecaster:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[key]()


def available() -> Dict[str, Callable[[], BaseForecaster]]:
    return dict(_REGISTRY)

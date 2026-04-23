"""Measure registry. Discovers external measures via entry points."""
from __future__ import annotations

import warnings
from importlib.metadata import entry_points

from .protocol import StereotypeMeasure
from .stub import StubMeasure

ENTRY_POINT_GROUP = "stereotype_bench.measures"

_registry: dict[str, type] = {"stub": StubMeasure}


def _load_plugins() -> None:
    eps = entry_points(group=ENTRY_POINT_GROUP)
    for ep in eps:
        if ep.name in _registry:
            continue  # built-ins win over entry-point shadows
        try:
            cls = ep.load()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Failed to load measure plugin {ep.name!r}: {exc}",
                stacklevel=2,
            )
            continue
        _registry[ep.name] = cls


_load_plugins()


def get_measure(name: str) -> StereotypeMeasure:
    if name not in _registry:
        raise KeyError(
            f"Unknown measure: {name!r}. Available: {list_measures()}"
        )
    return _registry[name]()


def list_measures() -> list[str]:
    return sorted(_registry)


__all__ = ["StereotypeMeasure", "get_measure", "list_measures", "ENTRY_POINT_GROUP"]

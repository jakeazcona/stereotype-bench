"""Protocol every stereotype measure must satisfy."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class StereotypeMeasure(Protocol):
    """A scalar stereotype-association measure over natural-language text.

    `score(text)` returns a float on a continuous axis whose endpoints are
    described by `axis_labels` = (low_label, high_label). Higher score
    indicates stronger association with `axis_labels[1]`.
    """

    name: str
    axis_labels: tuple[str, str]

    def score(self, text: str) -> float: ...

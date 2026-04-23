"""Built-in placeholder measure. Returns 0.0 for all inputs.

Useful for exercising the full pipeline without a real measure installed
(CI, demos, smoke tests). Real measures (e.g. gsa-core) register themselves
via the `stereotype_bench.measures` entry-point group.
"""
from __future__ import annotations

import warnings


class StubMeasure:
    name = "stub"
    axis_labels = ("low", "high")

    def __init__(self) -> None:
        warnings.warn(
            "StubMeasure returns 0.0 for all inputs. Install a real measure "
            "package (e.g. gsa-core) and select it in your experiment YAML.",
            stacklevel=2,
        )

    def score(self, text: str) -> float:  # noqa: ARG002
        return 0.0

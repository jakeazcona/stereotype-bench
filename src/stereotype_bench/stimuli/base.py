"""Stimulus types — the input shown to a model.

v1 uses TextStimulus only. ImageStimulus is reserved for the v2 VLM track
(showing models actual face images instead of text personas).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Stimulus(Protocol):
    kind: str


@dataclass
class TextStimulus:
    text: str
    kind: str = "text"


@dataclass
class ImageStimulus:
    """v2 seam — not used by v1 tasks."""

    image_path: str
    caption: str | None = None
    kind: str = "image"

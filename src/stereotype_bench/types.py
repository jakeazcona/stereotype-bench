"""Cross-module dataclasses kept here to avoid circular imports."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str

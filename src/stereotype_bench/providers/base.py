"""Provider abstraction. Adapters implement this Protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..types import Message


@dataclass
class GenerationResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None  # None if provider didn't return one
    raw: dict = field(default_factory=dict)


class ModelProvider(Protocol):
    provider_id: str

    def generate(
        self,
        model: str,
        messages: list[Message],
        **kwargs,
    ) -> GenerationResult: ...

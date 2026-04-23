"""Static pricing table loaded from configs/pricing.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

# Editable installs preserve the source layout, so this resolves to <repo>/configs.
DEFAULT_PRICING_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "configs" / "pricing.yaml"
)


class PricingTable:
    def __init__(self, data: dict) -> None:
        self.data = data or {}

    @classmethod
    def load(cls, path: Path | str | None = None) -> "PricingTable":
        p = Path(path) if path else DEFAULT_PRICING_PATH
        with p.open() as f:
            return cls(yaml.safe_load(f) or {})

    def cost_usd(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float | None:
        m = (self.data.get("models") or {}).get(model)
        if not m:
            return None
        per_million = 1_000_000.0
        return (input_tokens / per_million) * float(m.get("input", 0.0)) + (
            output_tokens / per_million
        ) * float(m.get("output", 0.0))

    def known_models(self) -> list[str]:
        return sorted((self.data.get("models") or {}).keys())

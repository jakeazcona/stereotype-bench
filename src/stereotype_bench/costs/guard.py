"""Hard budget guard - refuses calls past `budget_usd` unless force=True."""
from __future__ import annotations

from dataclasses import dataclass

from .db import CostDB


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetGuard:
    budget_usd: float
    db: CostDB
    force: bool = False

    def check(self, anticipated_usd: float = 0.0, run_id: str | None = None) -> None:
        if self.force:
            return
        current = self.db.total(run_id=run_id)
        if current + anticipated_usd > self.budget_usd:
            raise BudgetExceeded(
                f"Budget exceeded: spent {current:.4f} + anticipated "
                f"{anticipated_usd:.4f} USD > {self.budget_usd:.2f} cap "
                f"(run_id={run_id or 'all'}). Pass --force to override."
            )

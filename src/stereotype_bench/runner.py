"""Experiment orchestration."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .costs.db import CallRecord, CostDB
from .costs.guard import BudgetGuard
from .costs.pricing import PricingTable
from .measures import get_measure
from .providers.openrouter import OpenRouterProvider
from .tasks import get_task


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    measure: str
    task: str
    models: list[str]
    repetitions: int = 1
    budget_usd: float = 100.0
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        with Path(path).open() as f:
            data = yaml.safe_load(f) or {}
        known = {
            "name", "seed", "measure", "task",
            "models", "repetitions", "budget_usd",
        }
        extras = {k: v for k, v in data.items() if k not in known}
        return cls(
            name=data["name"],
            seed=int(data.get("seed", 0)),
            measure=data["measure"],
            task=data["task"],
            models=list(data["models"]),
            repetitions=int(data.get("repetitions", 1)),
            budget_usd=float(data.get("budget_usd", 100.0)),
            extras=extras,
        )


def run_experiment(
    config: ExperimentConfig,
    out_dir: Path | str,
    *,
    force: bool = False,
    cost_db: CostDB | None = None,
    pricing: PricingTable | None = None,
) -> dict:
    run_id = f"{config.name}-{uuid.uuid4().hex[:8]}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    db = cost_db or CostDB()
    pt = pricing if pricing is not None else PricingTable.load()
    guard = BudgetGuard(budget_usd=config.budget_usd, db=db, force=force)
    measure = get_measure(config.measure)
    task = get_task(config.task)
    provider = OpenRouterProvider()

    results: list[dict] = []
    for prompt in task.prompts():
        for model in config.models:
            for rep in range(config.repetitions):
                guard.check(run_id=run_id)
                result = provider.generate(model, prompt.messages)
                cost = result.cost_usd
                if cost is None:
                    cost = (
                        pt.cost_usd(
                            model,
                            result.input_tokens or 0,
                            result.output_tokens or 0,
                        )
                        or 0.0
                    )
                db.record(
                    CallRecord(
                        run_id=run_id,
                        provider=provider.provider_id,
                        model=model,
                        purpose="generation",
                        prompt_id=prompt.prompt_id,
                        input_tokens=result.input_tokens,
                        output_tokens=result.output_tokens,
                        cost_usd=float(cost),
                        metadata_json=json.dumps(prompt.metadata),
                    )
                )
                results.append(
                    {
                        "run_id": run_id,
                        "model": model,
                        "prompt_id": prompt.prompt_id,
                        "rep": rep,
                        "text": result.text,
                        "score": measure.score(result.text),
                        "metadata": prompt.metadata,
                    }
                )

    results_path = out / f"{run_id}.jsonl"
    with results_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return {
        "run_id": run_id,
        "results_path": str(results_path),
        "n_results": len(results),
        "total_cost_usd": db.total(run_id=run_id),
    }

"""Experiment orchestration with concurrent generation."""
from __future__ import annotations

import json
import threading
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .costs.db import CallRecord, CostDB
from .costs.guard import BudgetGuard
from .costs.pricing import PricingTable
from .measures import get_measure
from .providers.openrouter import OpenRouterProvider
from .tasks import get_task

# Stderr console so the progress bar doesn't pollute stdout (which CLI commands
# may use for piping).
_console = Console(stderr=True)


@dataclass
class ExperimentConfig:
    name: str
    seed: int
    measure: str
    task: str
    models: list[str]
    repetitions: int = 1
    budget_usd: float = 100.0
    concurrency: int = 8
    generation_params: dict = field(default_factory=dict)
    task_params: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "ExperimentConfig":
        with Path(path).open() as f:
            data = yaml.safe_load(f) or {}
        known = {
            "name", "seed", "measure", "task",
            "models", "repetitions", "budget_usd", "concurrency",
            "generation_params", "task_params",
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
            concurrency=int(data.get("concurrency", 8)),
            generation_params=dict(data.get("generation_params") or {}),
            task_params=dict(data.get("task_params") or {}),
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
    """Run an experiment concurrently, writing results incrementally."""
    run_id = f"{config.name}-{uuid.uuid4().hex[:8]}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    db = cost_db or CostDB()
    pt = pricing if pricing is not None else PricingTable.load()
    guard = BudgetGuard(budget_usd=config.budget_usd, db=db, force=force)
    measure = get_measure(config.measure)

    # Build the task with engine-managed defaults plus user task_params.
    task_kwargs = {
        "measure": measure,
        "repetitions": config.repetitions,
        "seed": config.seed,
        **config.task_params,
    }
    task = get_task(config.task, **task_kwargs)
    provider = OpenRouterProvider()

    # Pre-flight budget check (cheap to check before kicking off threads).
    guard.check(run_id=run_id)

    prompts = list(task.prompts())
    trials = [(p, m) for p in prompts for m in config.models]
    n_trials = len(trials)
    _console.log(
        f"[bold]Run {run_id}[/]: {n_trials} trials "
        f"({len(prompts)} prompts × {len(config.models)} models, concurrency={config.concurrency})"
    )

    results_path = out / f"{run_id}.jsonl"
    write_lock = threading.Lock()
    n_success = 0
    failures: list[tuple[str, str, str]] = []  # (model, prompt_id, error)

    def run_trial(trial: tuple) -> dict | None:
        prompt, model = trial
        try:
            result = provider.generate(
                model, prompt.messages, **config.generation_params
            )
            cleaned = (
                task.clean_output(result.text)
                if hasattr(task, "clean_output")
                else result.text
            )
            score = measure.score(cleaned)
        except Exception as exc:
            return {
                "_failure": True,
                "model": model,
                "prompt_id": prompt.prompt_id,
                "error": f"{type(exc).__name__}: {exc}",
            }

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

        return {
            "_failure": False,
            "row": {
                "run_id": run_id,
                "model": model,
                "prompt_id": prompt.prompt_id,
                "rep": prompt.metadata.get("rep", 0),
                "text_raw": result.text,
                "text": cleaned,
                "score": score,
                "metadata": prompt.metadata,
            },
            "cost_record": CallRecord(
                run_id=run_id,
                provider=provider.provider_id,
                model=model,
                purpose="generation",
                prompt_id=prompt.prompt_id,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=float(cost),
                metadata_json=json.dumps(prompt.metadata),
            ),
        }

    with results_path.open("w") as out_file, ThreadPoolExecutor(
        max_workers=max(1, int(config.concurrency))
    ) as ex:
        futures = [ex.submit(run_trial, t) for t in trials]
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("·"),
            TimeRemainingColumn(),
            console=_console,
            transient=False,
        ) as progress:
            pid = progress.add_task("Running", total=n_trials)
            for fut in as_completed(futures):
                result = fut.result()
                if result is None:
                    pass
                elif result["_failure"]:
                    failures.append(
                        (result["model"], result["prompt_id"], result["error"])
                    )
                else:
                    with write_lock:
                        out_file.write(json.dumps(result["row"]) + "\n")
                        out_file.flush()
                        db.record(result["cost_record"])
                    n_success += 1
                progress.update(pid, advance=1)

    if failures:
        _console.log(
            f"[yellow]{len(failures)} of {n_trials} trials failed.[/yellow]"
        )
        by_err = Counter(err[:120] for _m, _p, err in failures)
        for msg, n in by_err.most_common(5):
            _console.log(f"  [{n}x] {msg}")

    return {
        "run_id": run_id,
        "results_path": str(results_path),
        "n_results": n_success,
        "n_failures": len(failures),
        "n_trials": n_trials,
        "total_cost_usd": db.total(run_id=run_id),
    }

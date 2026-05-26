"""Experiment orchestration with concurrent generation."""
from __future__ import annotations

import contextlib
import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .costs.db import CallRecord, CostDB
from .costs.guard import BudgetGuard
from .costs.pricing import PricingTable
from .measures import get_measure
from .progress import ProgressTracker
from .providers.openrouter import OpenRouterProvider
from .tasks import get_task

_log = logging.getLogger("stereotype_bench.runner")


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
    # Models listed here run in a separate, small thread-pool so their upstream
    # rate-limits don't starve the main pool. Empty list = single pool.
    slow_models: list[str] = field(default_factory=list)
    slow_concurrency: int = 3
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
            "slow_models", "slow_concurrency",
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
            slow_models=list(data.get("slow_models") or []),
            slow_concurrency=int(data.get("slow_concurrency", 3)),
            generation_params=dict(data.get("generation_params") or {}),
            task_params=dict(data.get("task_params") or {}),
            extras=extras,
        )


def _load_completed_pairs(results_path: Path) -> set[tuple[str, str]]:
    """Read an existing JSONL and return the set of (prompt_id, model) rows."""
    pairs: set[tuple[str, str]] = set()
    if not results_path.exists():
        return pairs
    broken = 0
    with results_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                pairs.add((row["prompt_id"], row["model"]))
            except (json.JSONDecodeError, KeyError):
                broken += 1
    if broken:
        _log.warning(
            "resume: %d unparseable/malformed rows in %s (ignored)",
            broken, results_path,
        )
    return pairs


def run_experiment(
    config: ExperimentConfig,
    out_dir: Path | str,
    *,
    force: bool = False,
    cost_db: CostDB | None = None,
    pricing: PricingTable | None = None,
    resume_run_id: str | None = None,
) -> dict:
    """Run an experiment concurrently, writing results incrementally.

    If `resume_run_id` is provided, append to `<out_dir>/<resume_run_id>.jsonl`
    and skip any (prompt_id, model) pairs already present in that file.
    """
    if resume_run_id:
        run_id = resume_run_id
    else:
        run_id = f"{config.name}-{uuid.uuid4().hex[:8]}"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    db = cost_db or CostDB()
    pt = pricing if pricing is not None else PricingTable.load()
    guard = BudgetGuard(budget_usd=config.budget_usd, db=db, force=force)
    measure = get_measure(config.measure)

    task_kwargs = {
        "measure": measure,
        "repetitions": config.repetitions,
        "seed": config.seed,
        **config.task_params,
    }
    task = get_task(config.task, **task_kwargs)
    provider = OpenRouterProvider()

    guard.check(run_id=run_id)

    prompts = list(task.prompts())
    all_trials = [(p, m) for p in prompts for m in config.models]
    n_total = len(all_trials)

    results_path = out / f"{run_id}.jsonl"
    completed_pairs: set[tuple[str, str]] = set()
    if resume_run_id:
        completed_pairs = _load_completed_pairs(results_path)
        _log.info(
            "resume: found %d completed (prompt_id, model) pairs in %s",
            len(completed_pairs), results_path,
        )

    slow_set = set(config.slow_models or [])
    trials = [
        (p, m) for (p, m) in all_trials
        if (p.prompt_id, m) not in completed_pairs
    ]
    fast_trials = [(p, m) for (p, m) in trials if m not in slow_set]
    slow_trials = [(p, m) for (p, m) in trials if m in slow_set]
    n_trials = len(trials)
    n_skipped = n_total - n_trials

    _log.info(
        "run=%s mode=%s total=%d skip=%d todo=%d (fast=%d slow=%d) "
        "prompts=%d models=%d concurrency=%d slow_concurrency=%d slow_models=%s",
        run_id,
        "resume" if resume_run_id else "fresh",
        n_total, n_skipped, n_trials,
        len(fast_trials), len(slow_trials),
        len(prompts), len(config.models),
        config.concurrency, config.slow_concurrency,
        sorted(slow_set) or "[]",
    )
    if n_trials == 0:
        _log.info("nothing to do — all trials already present in %s", results_path)
        return {
            "run_id": run_id,
            "results_path": str(results_path),
            "n_results": 0,
            "n_failures": 0,
            "n_trials": n_total,
            "total_cost_usd": db.total(run_id=run_id),
        }

    write_lock = threading.Lock()
    tracker = ProgressTracker(n_total=n_total, n_already_done=n_skipped)

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
                "error_type": type(exc).__name__,
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
            "model": model,
            "prompt_id": prompt.prompt_id,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cost_usd": float(cost),
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

    # Append mode when resuming; write mode for a fresh run (we've already
    # verified there's nothing to preserve when not resuming).
    file_mode = "a" if resume_run_id else "w"
    with contextlib.ExitStack() as stack:
        out_file = stack.enter_context(results_path.open(file_mode))
        fast_ex = stack.enter_context(
            ThreadPoolExecutor(max_workers=max(1, int(config.concurrency)))
        )
        slow_ex = None
        if slow_trials:
            slow_ex = stack.enter_context(
                ThreadPoolExecutor(max_workers=max(1, int(config.slow_concurrency)))
            )
        futures = [fast_ex.submit(run_trial, t) for t in fast_trials]
        if slow_ex is not None:
            futures.extend(slow_ex.submit(run_trial, t) for t in slow_trials)
        for fut in as_completed(futures):
            result = fut.result()
            if result is None:
                continue
            if result["_failure"]:
                tracker.update_failure(result["model"], result["error_type"])
            else:
                with write_lock:
                    out_file.write(json.dumps(result["row"]) + "\n")
                    out_file.flush()
                    db.record(result["cost_record"])
                tracker.update_success(
                    result["model"],
                    result["input_tokens"],
                    result["output_tokens"],
                    result["cost_usd"],
                )
            tracker.maybe_snapshot()

    tracker.final_summary()

    return {
        "run_id": run_id,
        "results_path": str(results_path),
        "n_results": tracker.succeeded,
        "n_failures": tracker.failed,
        "n_trials": n_total,
        "total_cost_usd": db.total(run_id=run_id),
    }

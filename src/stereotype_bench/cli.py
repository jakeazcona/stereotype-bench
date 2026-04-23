"""Typer-based CLI entry point."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from . import measures as _measures
from .costs.db import CostDB
from .costs.pricing import PricingTable
from .plots.means_ci import plot_means_ci
from .runner import ExperimentConfig, run_experiment
from .tasks import TASKS

# Auto-load .env from CWD or parent dirs so API keys (OPENROUTER_API_KEY,
# NVIDIA_API_KEY) are available without manual `export`. Does not override
# pre-set environment variables.
load_dotenv()

app = typer.Typer(
    help="Measure-agnostic LLM stereotype benchmark engine.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    config: Path = typer.Argument(..., exists=True, help="Path to experiment YAML"),
    out: Path = typer.Option(Path("./runs"), help="Output directory for results"),
    force: bool = typer.Option(False, "--force", help="Override budget guard"),
) -> None:
    """Run an experiment from a YAML config."""
    cfg = ExperimentConfig.from_yaml(config)
    console.print(
        f"[bold]Running:[/] {cfg.name}  (measure={cfg.measure}, task={cfg.task})"
    )
    summary = run_experiment(cfg, out, force=force)
    console.print(f"[green]Run complete:[/] {summary['run_id']}")
    console.print(
        f"  Results: {summary['results_path']}  ({summary['n_results']} rows)"
    )
    console.print(f"  Cost:    ${summary['total_cost_usd']:.4f}")


@app.command("list-measures")
def list_measures_cmd() -> None:
    """List registered stereotype measures."""
    names = _measures.list_measures()
    console.print(f"Available measures: [bold]{', '.join(names)}[/]")


@app.command("list-tasks")
def list_tasks_cmd() -> None:
    """List registered tasks."""
    console.print(f"Available tasks: [bold]{', '.join(sorted(TASKS))}[/]")


@app.command("list-models")
def list_models_cmd() -> None:
    """List models present in the pricing table."""
    pt = PricingTable.load()
    for m in pt.known_models():
        console.print(f"  {m}")


@app.command()
def plot(
    run_id: str = typer.Argument(
        ...,
        help="Run ID — JSONL filename stem (without .jsonl) under --runs-dir.",
    ),
    runs_dir: Path = typer.Option(
        Path("./runs"), help="Directory containing run JSONL files"
    ),
    out: Optional[Path] = typer.Option(
        None, help="Output figure path (defaults to runs/<run_id>.png)"
    ),
    title: Optional[str] = typer.Option(None, help="Figure title"),
) -> None:
    """Mean ± 95% CI bar chart for a completed run (one subplot per variant)."""
    jsonl = runs_dir / f"{run_id}.jsonl"
    if not jsonl.is_file():
        console.print(f"[red]Not found:[/] {jsonl}")
        raise typer.Exit(code=2)
    fig_path = out if out else runs_dir / f"{run_id}.png"
    plot_means_ci(jsonl, fig_path, title=title or f"GSA: {run_id}")
    console.print(f"[green]Wrote:[/] {fig_path}")


@app.command()
def costs(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Filter to one run_id"),
    csv: Optional[Path] = typer.Option(None, "--csv", help="Export to CSV at this path"),
) -> None:
    """Show cumulative API spend; optionally export CSV (e.g. for reimbursement)."""
    db = CostDB()
    rows = db.by_model(run_id=run_id)
    table = Table(title=f"Spend ({'run=' + run_id if run_id else 'all runs'})")
    table.add_column("Model")
    table.add_column("Calls", justify="right")
    table.add_column("USD", justify="right")
    for row in rows:
        table.add_row(row["model"], str(row["n"]), f"${row['cost']:.4f}")
    console.print(table)
    console.print(f"[bold]Total:[/] ${db.total(run_id=run_id):.4f}")
    if csv:
        n = db.to_csv(csv, run_id=run_id)
        console.print(f"[green]CSV exported:[/] {csv}  ({n} rows)")


if __name__ == "__main__":
    app()

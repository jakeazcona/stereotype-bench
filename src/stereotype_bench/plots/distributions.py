"""Per-model score distribution plots."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(path: Path | str) -> pd.DataFrame:
    rows: list[dict] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def plot_score_distribution(
    results_jsonl: Path | str,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    df = load_results(results_jsonl)
    if df.empty:
        raise ValueError(f"No results in {results_jsonl}")

    fig, ax = plt.subplots(figsize=(10, 6))
    models = sorted(df["model"].unique())
    data = [df.loc[df["model"] == m, "score"].to_numpy() for m in models]
    ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Stereotype-association score")
    ax.set_title(title or f"Score distribution - {Path(results_jsonl).stem}")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out

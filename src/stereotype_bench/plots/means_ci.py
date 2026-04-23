"""Per-(model, variant, gender) means with 95% CI bar charts.

Loads run JSONL produced by `stereotype-bench run`, aggregates scores into
mean ± SE, and renders a multi-subplot bar chart (one subplot per variant,
two bars per model for male/female).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Two-sided 95% CI z-score from normal approximation. With n>=30 per cell
# (which 100 reps × 25 personas trivially exceeds), this is fine.
_Z_95 = 1.96


def load_results(jsonl_path: Path | str) -> pd.DataFrame:
    """Flatten run JSONL into a DataFrame keyed for grouping."""
    rows = [
        json.loads(line)
        for line in Path(jsonl_path).read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"no results in {jsonl_path}")
    out = []
    for r in rows:
        meta = r.get("metadata", {}) or {}
        out.append(
            {
                "run_id": r.get("run_id"),
                "model": r.get("model"),
                "variant": meta.get("variant", "plain"),
                "gender": meta.get("gender"),
                "name": meta.get("name"),
                "age": meta.get("age"),
                "score": r.get("score"),
            }
        )
    return pd.DataFrame(out)


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (model, variant, gender) -> mean, std, count, sem, 95% CI bounds."""
    grouped = df.groupby(["model", "variant", "gender"], dropna=False)["score"]
    out = grouped.agg(["mean", "std", "count"]).reset_index()
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    out["ci_low"] = out["mean"] - _Z_95 * out["sem"]
    out["ci_high"] = out["mean"] + _Z_95 * out["sem"]
    return out


def _short_model_label(model_id: str, max_len: int = 22) -> str:
    """Trim provider/long-name down to something readable on x-axis."""
    short = model_id.split("/", 1)[-1]
    if len(short) > max_len:
        short = short[: max_len - 1] + "…"
    return short


def plot_means_ci(
    jsonl_path: Path | str,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    """Bar chart with 95% CI: one subplot per variant, two bars per model (gender)."""
    df = load_results(jsonl_path)
    stats_df = compute_stats(df)

    variants = sorted(df["variant"].dropna().unique())
    models = sorted(df["model"].dropna().unique())
    genders = ["male", "female"]
    colors = {"male": "#3b82f6", "female": "#ec4899"}

    n_v = len(variants)
    cols = min(2, max(1, n_v))
    rows = (n_v + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(7 * cols, 4.2 * rows), squeeze=False, sharey=True
    )

    for ax_idx, variant in enumerate(variants):
        ax = axes[ax_idx // cols][ax_idx % cols]
        sub = stats_df[stats_df["variant"] == variant]
        x = np.arange(len(models))
        width = 0.38

        for gi, gender in enumerate(genders):
            gsub = (
                sub[sub["gender"] == gender]
                .set_index("model")
                .reindex(models)
            )
            means = gsub["mean"].fillna(0).to_numpy()
            err_low = (gsub["mean"] - gsub["ci_low"]).fillna(0).to_numpy()
            err_high = (gsub["ci_high"] - gsub["mean"]).fillna(0).to_numpy()
            n_per_cell = gsub["count"].fillna(0).astype(int).to_numpy()

            ax.bar(
                x + (gi - 0.5) * width,
                means,
                width,
                yerr=[err_low, err_high],
                label=f"{gender} (n={int(n_per_cell.max() or 0)})",
                color=colors[gender],
                capsize=3,
                edgecolor="black",
                linewidth=0.4,
            )

        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"variant: {variant}")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_short_model_label(m) for m in models], rotation=30, ha="right"
        )
        if ax_idx % cols == 0:
            ax.set_ylabel("GSA score (mean ± 95% CI)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Hide unused subplots if variant count doesn't fill the grid.
    for idx in range(n_v, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, y=1.0)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out

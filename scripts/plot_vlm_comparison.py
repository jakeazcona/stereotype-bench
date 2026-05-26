"""Render the LLM-vs-VLM GSA comparison figure.

Loads the existing LLM run JSONL (gsa_first_impression_v1) plus the 6 scored
VLM JSONLs, filters to `plain` + `three_traits`, and draws a 2-subplot bar
chart of mean GSA score by gender. Borrows the stats helper from the engine's
`plots.means_ci` but uses an explicit model ordering so the 3 VLMs appear
grouped to the right of the 7 LLMs in pedagogical order.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stereotype_bench.plots.means_ci import compute_stats, load_results

ENGINE_DIR = Path.home() / "projects" / "stereotype-bench"
LLM_JSONL = ENGINE_DIR / "runs" / "gsa_first_impression_v1-f636d8da.jsonl"
VLM_DIR = ENGINE_DIR / "runs" / "vlm"
VLM_FILES = [
    "preds_base_describe30.scored.jsonl",
    "preds_base_three_traits_open.scored.jsonl",
    "preds_ft_baseline_describe30.scored.jsonl",
    "preds_ft_baseline_three_traits_open.scored.jsonl",
    "preds_ft_stereo_describe30.scored.jsonl",
    "preds_ft_stereo_three_traits_open.scored.jsonl",
]
OUT_PNG = ENGINE_DIR / "runs" / "vlm_vs_llm_first_impression.png"

VARIANTS = ["plain", "three_traits"]

# LLM models alphabetical (matches original figure) + VLMs in pedagogical order.
LLM_ORDER = [
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4.5",
    "google/gemini-2.5-flash",
    "meta-llama/llama-3.3-70b-instruct",
    "moonshotai/kimi-k2",
    "openai/gpt-4o-mini",
    "x-ai/grok-4-fast",
]
VLM_ORDER = [
    "qwen-2.5-vl/untuned",
    "qwen-2.5-vl/ft-baseline",
    "qwen-2.5-vl/ft-stereo",
]
MODEL_ORDER = LLM_ORDER + VLM_ORDER

_Z_95 = 1.96


def _short_label(model_id: str, max_len: int = 22) -> str:
    short = model_id.split("/", 1)[-1]
    if len(short) > max_len:
        short = short[: max_len - 1] + "…"
    return short


def load_all() -> pd.DataFrame:
    """Concatenate LLM run + 6 scored VLM files into one long DataFrame."""
    frames = [load_results(LLM_JSONL)]
    for f in VLM_FILES:
        frames.append(load_results(VLM_DIR / f))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["variant"].isin(VARIANTS)]
    df = df[df["gender"].isin(["male", "female"])]
    return df


def plot_comparison(df: pd.DataFrame, out_path: Path) -> Path:
    stats_df = compute_stats(df)
    models = MODEL_ORDER
    genders = ["male", "female"]
    colors = {"male": "#3b82f6", "female": "#ec4899"}

    fig, axes = plt.subplots(
        1, 2, figsize=(7 * 2, 5.0), squeeze=False, sharey=True
    )

    for ax_idx, variant in enumerate(VARIANTS):
        ax = axes[0][ax_idx]
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
                label=f"{gender} (n_max={int(n_per_cell.max() or 0)})",
                color=colors[gender],
                capsize=3,
                edgecolor="black",
                linewidth=0.4,
            )

        # Visual divider between LLM block and VLM block.
        ax.axvline(len(LLM_ORDER) - 0.5, color="black", linestyle="--", linewidth=0.6, alpha=0.5)

        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"variant: {variant}")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_short_label(m) for m in models], rotation=30, ha="right"
        )
        if ax_idx == 0:
            ax.set_ylabel("GSA score (mean ± 95% CI)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle(
        "GSA first-impression scores: 7 LLMs (text personas) vs 3 Qwen-2.5-VL variants (CFD-W faces)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> int:
    df = load_all()
    print(f"Loaded {len(df)} rows across {df['model'].nunique()} models, "
          f"{df['variant'].nunique()} variants.")
    print("\nPer-cell counts:")
    print(df.groupby(["model", "variant", "gender"]).size().unstack(fill_value=0))
    out = plot_comparison(df, OUT_PNG)
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

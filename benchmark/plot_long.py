"""Generate per-bucket comparison charts from long-context benchmark results.

Usage::

    cd qwen-long-context/benchmark
    python plot_long.py                             # saves long_results/infer_long_comparison.png
    python plot_long.py --results-dir long_results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE         = Path(__file__).resolve().parent
RESULTS_DIR  = HERE / "long_results"
BUCKET_ORDER = ["short", "medium_short", "medium_long", "long"]

MODEL_ORDER  = ["qwen-base", "supercoder", "lazy-morph"]
MODEL_LABELS = {
    "qwen-base":   "Qwen2.5-Coder 7B",
    "supercoder":  "SuperCoder (exp1)",
    "lazy-morph":  "Lazy+Morph",
}
COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def load_data(results_dir: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    summary = pd.read_csv(results_dir / "infer_summary_by_bucket.csv")

    details: dict[str, pd.DataFrame] = {}
    for p in sorted(results_dir.glob("infer_results_*.csv")):
        tag = p.stem.removeprefix("infer_results_")
        details[tag] = pd.read_csv(p)

    return summary, details


def _grouped_bar(ax, summary: pd.DataFrame, models: list[str], value_col: str,
                 ylabel: str, title: str, fmt: str = "%.2f") -> None:
    x = np.arange(len(BUCKET_ORDER))
    width = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        y = []
        for b in BUCKET_ORDER:
            sub = summary[(summary["model"] == m) & (summary["bucket"] == b)]
            val = float(sub[value_col].iloc[0]) if len(sub) and sub[value_col].iloc[0] != "" else 0.0
            y.append(val)
        offset = (i - (len(models) - 1) / 2) * width
        color = COLORS[MODEL_ORDER.index(m)] if m in MODEL_ORDER else "#888888"
        label = MODEL_LABELS.get(m, m)
        bars = ax.bar(x + offset, y, width, color=color, label=label,
                      edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt=fmt, padding=2, fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_ORDER, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def plot(results_dir: Path, out: Path) -> None:
    summary, details = load_data(results_dir)
    models = [m for m in MODEL_ORDER if m in summary["model"].values]
    if not models:
        models = list(summary["model"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Long-Context Benchmark -- CodeNet Balanced (by C code length bucket)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # 1. Compile rate by bucket
    _grouped_bar(axes[0, 0], summary, models, "compile_rate",
                 "fraction", "Compile Rate by Bucket")

    # 2. Test pass rate by bucket
    _grouped_bar(axes[0, 1], summary, models, "test_pass_rate",
                 "fraction", "All-Tests Pass Rate by Bucket")

    # 3. Mean correctness by bucket
    _grouped_bar(axes[0, 2], summary, models, "mean_correctness",
                 "score", "Mean Correctness by Bucket", fmt="%.3f")

    # 4. Geo-mean speedup by bucket
    has_speedup = "geo_mean_floor1" in summary.columns
    if has_speedup:
        num = summary.copy()
        num["geo_mean_floor1"] = pd.to_numeric(num["geo_mean_floor1"], errors="coerce").fillna(1.0)
        _grouped_bar(axes[1, 0], num, models, "geo_mean_floor1",
                     "geo-mean speedup", "Geo-Mean Speedup by Bucket (paper metric)", fmt="%.3fx")
        axes[1, 0].axhline(1.0, color="grey", lw=0.8, ls="--")
    else:
        axes[1, 0].text(0.5, 0.5, "No speedup data\n(re-run with --do-speedup)",
                        ha="center", va="center", transform=axes[1, 0].transAxes,
                        fontsize=10, color="grey")
        axes[1, 0].set_title("Geo-Mean Speedup by Bucket", fontweight="bold")

    # 5. Copy rate by bucket
    _grouped_bar(axes[1, 1], summary, models, "copy_rate",
                 "fraction", "Copy Rate by Bucket (lower is better)", fmt="%.2f")

    # 6. Correctness distribution by bucket (box plot)
    ax = axes[1, 2]
    if details:
        x = np.arange(len(BUCKET_ORDER))
        width = 0.8 / max(1, len(models))
        positions_all = []
        data_all = []
        colors_all = []

        for bi, b in enumerate(BUCKET_ORDER):
            for mi, m in enumerate(models):
                if m not in details:
                    continue
                sub = details[m]
                if "bucket" in sub.columns:
                    vals = sub[sub["bucket"] == b]["correctness"].dropna().tolist()
                else:
                    vals = []
                if not vals:
                    vals = [np.nan]
                data_all.append(vals)
                offset = (mi - (len(models) - 1) / 2) * width
                positions_all.append(bi + offset)
                colors_all.append(COLORS[MODEL_ORDER.index(m)] if m in MODEL_ORDER else "#888888")

        if data_all:
            bp = ax.boxplot(
                data_all, positions=positions_all, widths=width * 0.85,
                patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5),
                flierprops=dict(marker="o", markersize=2, alpha=0.3),
            )
            for patch, color in zip(bp["boxes"], colors_all):
                patch.set_facecolor(color)
                patch.set_alpha(0.65)
            ax.set_xticks(np.arange(len(BUCKET_ORDER)))
            ax.set_xticklabels(BUCKET_ORDER, fontsize=9)
            handles = [plt.Rectangle((0, 0), 1, 1,
                       facecolor=COLORS[MODEL_ORDER.index(m)] if m in MODEL_ORDER else "#888",
                       alpha=0.65, label=MODEL_LABELS.get(m, m))
                       for m in models]
            ax.legend(handles=handles, fontsize=8)
    ax.set_title("Correctness Distribution by Bucket", fontweight="bold")
    ax.set_ylabel("correctness score")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (args.results_dir / "infer_long_comparison.png")
    plot(args.results_dir, out)


if __name__ == "__main__":
    main()

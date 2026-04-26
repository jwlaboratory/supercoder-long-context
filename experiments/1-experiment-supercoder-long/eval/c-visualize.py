"""Step C — Visualize per-bucket results (runs on your PC).

Reads every `infer_<tag>__benchmarked.csv` produced by step B and produces:

    results/figures/compile_rate_by_bucket.png
    results/figures/pass_rate_by_bucket.png
    results/figures/mean_speedup_by_bucket.png
    results/figures/speedup_distribution.png
    results/summary_by_bucket.csv

Summary prints a side-by-side table per bucket per model.

Requirements:
    pip install pandas matplotlib numpy

Usage:
    cd qwen-long-context/1-experiment-supercoder-long/eval
    uv run python c-visualize.py
    uv run python c-visualize.py --results-dir results --out-dir results/figures
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE         = Path(__file__).resolve().parent
RESULTS_DIR  = HERE / "results"
BUCKET_ORDER = ["short", "medium_short", "medium_long", "long"]


def _load_all(results_dir: Path) -> dict[str, pd.DataFrame]:
    csv.field_size_limit(sys.maxsize)
    csvs = sorted(results_dir.glob("infer_*__benchmarked.csv"))
    if not csvs:
        raise SystemExit(
            f"No *__benchmarked.csv found in {results_dir}. "
            f"Run b-benchmark-local.py first."
        )
    out: dict[str, pd.DataFrame] = {}
    for p in csvs:
        tag = p.stem.removeprefix("infer_").removesuffix("__benchmarked")
        df  = pd.read_csv(p, engine="python")
        for col in ("compile_gen_ok", "all_tests_pass", "is_copy"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().isin({"true", "1", "yes"})
        for col in ("test_pass_rate", "speedup", "unopt_mean_ms", "gen_mean_ms"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "bucket" in df.columns:
            df["bucket"] = pd.Categorical(df["bucket"], categories=BUCKET_ORDER, ordered=True)
        out[tag] = df
        print(f"  [{tag}] {len(df)} rows")
    return out


# ── Aggregation ──────────────────────────────────────────────────────────────

def _per_bucket_summary(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for tag, df in dfs.items():
        for b in BUCKET_ORDER:
            sub = df[df["bucket"] == b]
            n = len(sub)
            if n == 0:
                continue
            sp_vals = sub["speedup"].dropna() if "speedup" in sub.columns else pd.Series([])
            rows.append({
                "model":           tag,
                "bucket":          b,
                "n":               n,
                "compile_rate":    round(sub["compile_gen_ok"].mean(), 3),
                "all_pass_rate":   round(sub["all_tests_pass"].mean(), 3),
                "mean_test_pass":  round(sub["test_pass_rate"].mean(), 3),
                "is_copy_rate":    round(sub.get("is_copy", pd.Series([False]*n)).mean(), 3),
                "n_with_speedup":  int(len(sp_vals)),
                "mean_speedup":    round(float(sp_vals.mean()), 4) if len(sp_vals) else "",
                "median_speedup":  round(float(sp_vals.median()), 4) if len(sp_vals) else "",
                "max_speedup":     round(float(sp_vals.max()), 4)    if len(sp_vals) else "",
            })
    return pd.DataFrame(rows)


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _grouped_bar(ax, summary: pd.DataFrame, value_col: str, ylabel: str, title: str) -> None:
    models = list(summary["model"].unique())
    x      = np.arange(len(BUCKET_ORDER))
    width  = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        y = []
        for b in BUCKET_ORDER:
            sub = summary[(summary["model"] == m) & (summary["bucket"] == b)]
            y.append(float(sub[value_col].iloc[0]) if len(sub) else 0.0)
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, y, width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def _box_speedup(ax, dfs: dict[str, pd.DataFrame]) -> None:
    """Boxplot of speedup per bucket grouped by model."""
    models = list(dfs.keys())
    positions = []
    data = []
    labels = []
    group_w = 0.8
    width  = group_w / max(1, len(models))
    for bi, b in enumerate(BUCKET_ORDER):
        for mi, m in enumerate(models):
            sub = dfs[m]
            if "bucket" not in sub.columns or "speedup" not in sub.columns:
                continue
            vals = sub[sub["bucket"] == b]["speedup"].dropna().values
            if len(vals) == 0:
                vals = np.array([np.nan])
            data.append(vals)
            positions.append(bi + (mi - (len(models) - 1) / 2) * width)
            labels.append(f"{b}\n{m}" if mi == 0 else m)
    bp = ax.boxplot(data, positions=positions, widths=width * 0.9, patch_artist=True,
                    showmeans=True, meanline=True)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(models))))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(models)])
        patch.set_alpha(0.6)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(np.arange(len(BUCKET_ORDER)))
    ax.set_xticklabels(BUCKET_ORDER)
    ax.set_ylabel("speedup (unopt_ms / gen_ms)")
    ax.set_title("Speedup distribution by bucket (higher = faster; 1.0 = no gain)")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.6, label=m)
               for i, m in enumerate(models)]
    ax.legend(handles=handles)
    ax.grid(axis="y", linestyle=":", alpha=0.5)


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--out-dir",     type=Path, default=None,
                   help="Figure output directory (default: <results-dir>/figures).")
    args = p.parse_args()

    fig_dir = args.out_dir or (args.results_dir / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    dfs = _load_all(args.results_dir)
    summary = _per_bucket_summary(dfs)
    summary_path = args.results_dir / "summary_by_bucket.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary → {summary_path}")
    print(summary.to_string(index=False))

    # Compile rate
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bar(ax, summary, "compile_rate",
                 "fraction compiles OK", "Compile rate by bucket")
    fig.tight_layout()
    fig.savefig(fig_dir / "compile_rate_by_bucket.png", dpi=150)
    plt.close(fig)

    # All-tests-pass rate
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bar(ax, summary, "all_pass_rate",
                 "fraction all-tests-pass", "All-tests-pass rate by bucket")
    fig.tight_layout()
    fig.savefig(fig_dir / "pass_rate_by_bucket.png", dpi=150)
    plt.close(fig)

    # Mean speedup (where measurable)
    if (summary["mean_speedup"] != "").any():
        num = summary.copy()
        num["mean_speedup"] = pd.to_numeric(num["mean_speedup"], errors="coerce").fillna(0)
        fig, ax = plt.subplots(figsize=(8, 5))
        _grouped_bar(ax, num, "mean_speedup",
                     "mean speedup (unopt / gen)", "Mean speedup by bucket (passes only)")
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
        fig.tight_layout()
        fig.savefig(fig_dir / "mean_speedup_by_bucket.png", dpi=150)
        plt.close(fig)

        # Distribution boxplot
        fig, ax = plt.subplots(figsize=(10, 6))
        _box_speedup(ax, dfs)
        fig.tight_layout()
        fig.savefig(fig_dir / "speedup_distribution.png", dpi=150)
        plt.close(fig)

    print(f"\nFigures → {fig_dir}")


if __name__ == "__main__":
    main()

"""Generate comparison charts from ``infer_results_*.csv`` and ``infer_summary.csv``.

Usage::

    cd qwen-long-context/benchmark
    python plot.py                     # saves infer_comparison.png
    python plot.py --out my_chart.png
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

MODEL_ORDER  = ["qwen-base", "supercoder", "lazy-morph"]
MODEL_LABELS = {
    "qwen-base":   "Qwen2.5-Coder\n7B-Instruct",
    "supercoder":  "SuperCoder\n(exp1 @ step 420)",
    "lazy-morph":  "Lazy+Morph\n(train2-lazy-supercoder)",
}
COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def load_data(here: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    details: dict[str, pd.DataFrame] = {}
    for p in sorted(here.glob("infer_results_*.csv")):
        tag = p.stem.removeprefix("infer_results_")
        details[tag] = pd.read_csv(p)

    if details:
        summary = _build_summary(details)
    else:
        summary = pd.read_csv(here / "infer_summary.csv")
        summary = summary.drop_duplicates("model", keep="last")

    summary["model_order"] = summary["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    summary["model_order"] = summary["model_order"].fillna(len(MODEL_ORDER)).astype(int)
    summary = summary.sort_values("model_order").reset_index(drop=True)

    return summary, details


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _numeric_series(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _build_summary(details: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model, df in details.items():
        n = len(df)
        if n == 0:
            continue

        compiled = _bool_series(df["compiled"]) if "compiled" in df.columns else pd.Series([False] * n)
        tests_pass = _bool_series(df["tests_pass"]) if "tests_pass" in df.columns else pd.Series([False] * n)
        correctness = _numeric_series(df, "correctness", -1.0).fillna(-1.0)
        raw_speedup = _numeric_series(df, "raw_speedup", np.nan)
        sp_vals = raw_speedup.dropna()

        effective_speedup = _numeric_series(df, "effective_speedup", np.nan)
        effective_speedup = effective_speedup.fillna(raw_speedup.where(tests_pass, 1.0)).fillna(1.0)
        speedup_floor1 = _numeric_series(df, "speedup_floor1", np.nan)
        speedup_floor1 = speedup_floor1.fillna(effective_speedup).clip(lower=1.0).fillna(1.0)
        geo_floor1 = float(np.exp(np.mean(np.log(speedup_floor1)))) if len(speedup_floor1) else 1.0

        status_counts = df["status"].value_counts().to_dict() if "status" in df.columns else {}
        copy_rate = _bool_series(df["is_copy"]).mean() if "is_copy" in df.columns else 0.0

        row = {
            "model": model,
            "n_samples": n,
            "compile_rate": round(float(compiled.mean()), 3),
            "test_pass_rate": round(float(tests_pass.mean()), 3),
            "mean_correctness": round(float(correctness.mean()), 3),
            "n_speedup_measured": int(len(sp_vals)),
            "mean_speedup": round(float(sp_vals.mean()), 4) if len(sp_vals) else np.nan,
            "max_speedup": round(float(sp_vals.max()), 4) if len(sp_vals) else np.nan,
            "mean_effective_speedup": round(float(effective_speedup.mean()), 4),
            "paper_avg_speedup": round(max(1.0, float(speedup_floor1.mean())), 4),
            "geo_mean_speedup_floor1": round(max(1.0, geo_floor1), 4),
            "p25_speedup_floor1": round(float(np.percentile(speedup_floor1, 25)), 4),
            "p50_speedup_floor1": round(float(np.percentile(speedup_floor1, 50)), 4),
            "p75_speedup_floor1": round(float(np.percentile(speedup_floor1, 75)), 4),
            "copy_rate": round(float(copy_rate), 3),
            "status_breakdown": str(status_counts),
        }

        if "morph_called" in df.columns:
            morph_called = _bool_series(df["morph_called"])
            morph_success = _bool_series(df["morph_success"]) if "morph_success" in df.columns else pd.Series([False] * n)
            n_morph = int(morph_called.sum())
            row["morph_call_rate"] = round(n_morph / n, 3)
            row["morph_success_rate"] = round(int(morph_success.sum()) / max(n_morph, 1), 3)

        rows.append(row)

    return pd.DataFrame(rows)


def _parse_status_counts(raw: str) -> dict[str, int]:
    try:
        return ast.literal_eval(raw)
    except Exception:
        return {}


def _all_sample_speedups(df: pd.DataFrame) -> np.ndarray:
    """Return raw speedups for passing samples and 0x for every failure."""
    vals = pd.to_numeric(df["raw_speedup"], errors="coerce").fillna(0.0)
    return vals.to_numpy(dtype=float)


def _plot_all_sample_speedup_distribution(
    ax: plt.Axes,
    models: list[str],
    labels: list[str],
    palette: list[str],
    details: dict[str, pd.DataFrame],
) -> None:
    """Overlay per-model speedup histograms, keeping failures visible at 0x."""
    bins = np.array([0, 0.05, 0.5, 0.8, 1.0, 1.1, 1.25, 1.5, 2.0, 4.0, 8.0, 12.0])
    any_plotted = False
    for model, color, label in zip(models, palette, labels):
        if model not in details:
            continue
        vals = _all_sample_speedups(details[model])
        if len(vals) == 0:
            continue
        ax.hist(
            np.clip(vals, bins[0], bins[-1]),
            bins=bins,
            alpha=0.45,
            density=True,
            color=color,
            label=label.replace("\n", " "),
            edgecolor="white",
        )
        any_plotted = True

    ax.axvline(0, color="#d62728", lw=1.2, ls=":", label="failure = 0x")
    ax.axvline(1.0, color="grey", lw=0.8, ls="--", label="baseline (1x)")
    ax.set_xlim(-0.05, 2.0)
    ax.set_xlabel("speedup; failures plotted at 0x")
    ax.set_ylabel("density")
    if any_plotted:
        ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def plot(out: Path) -> None:
    summary, details = load_data(HERE)
    models  = list(summary["model"])
    labels  = [MODEL_LABELS.get(m, m) for m in models]
    palette = [
        COLORS[MODEL_ORDER.index(m)] if m in MODEL_ORDER else "#888888"
        for m in models
    ]
    x       = np.arange(len(models))
    bar_w   = 0.55

    fig, axes = plt.subplots(3, 3, figsize=(16, 15))
    fig.suptitle(
        "Lazy+Morph Benchmark -- sc_val (C + slow asm -> faster asm)",
        fontsize=14, fontweight="bold", y=1.01,
    )

    # 1. Compile rate
    ax = axes[0, 0]
    vals = summary["compile_rate"].tolist()
    bars = ax.bar(x, vals, width=bar_w, color=palette, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_title("Compile Rate", fontweight="bold")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("fraction")
    ax.axhline(1.0, color="grey", lw=0.6, ls="--")
    ax.grid(axis="y", alpha=0.3)

    # 2. Test pass rate
    ax = axes[0, 1]
    vals = summary["test_pass_rate"].tolist()
    bars = ax.bar(x, vals, width=bar_w, color=palette, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_title("All-Tests Pass Rate", fontweight="bold")
    ax.set_ylim(0, min(1.15, max(max(vals) * 1.5 + 0.05, 0.1)))
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("fraction")
    ax.grid(axis="y", alpha=0.3)

    # 3. Mean correctness
    ax = axes[0, 2]
    vals = summary["mean_correctness"].tolist()
    bar_colors = [p if v >= 0 else "#d9534f" for v, p in zip(vals, palette)]
    bars = ax.bar(x, vals, width=bar_w, color=bar_colors, edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_title("Mean Correctness Score\n(-1=compile fail, 1=all pass)", fontweight="bold")
    ax.set_ylim(min(vals) - 0.15, 1.1)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("score")
    ax.grid(axis="y", alpha=0.3)

    # 4. Outcome-adjusted speedup distribution
    ax = axes[1, 0]
    _plot_all_sample_speedup_distribution(ax, models, labels, palette, details)
    ax.set_title("Speedup + Failure Distribution\n(all samples; failures at 0x)", fontweight="bold")

    # 5. Correctness distribution (box plot)
    ax = axes[1, 1]
    box_data = []
    for model in models:
        if model in details:
            box_data.append(details[model]["correctness"].tolist())
        else:
            box_data.append([])

    bp = ax.boxplot(
        box_data, patch_artist=True, widths=0.5,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color); patch.set_alpha(0.75)

    ax.set_title("Correctness Score Distribution\n(-1=compile fail ... 1=all pass)", fontweight="bold")
    ax.set_xticks(range(1, len(models) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("correctness score")
    ax.axhline(0, color="grey", lw=0.6, ls="--")
    ax.grid(axis="y", alpha=0.3)

    # 6. Correctness histogram (compiled only)
    ax = axes[1, 2]
    bins = np.linspace(0, 1, 11)
    any_plotted = False
    for model, color, label in zip(models, palette, labels):
        if model not in details:
            continue
        compiled_df = details[model][details[model]["correctness"] > -1]
        passing = compiled_df[compiled_df["correctness"] >= 0]["correctness"]
        if len(passing) == 0:
            continue
        ax.hist(passing, bins=bins, alpha=0.55, color=color,
                label=label.replace("\n", " "), edgecolor="white")
        any_plotted = True

    ax.set_title("Correctness Distribution\n(compiled samples only)", fontweight="bold")
    ax.set_xlabel("fraction of tests passed")
    ax.set_ylabel("count")
    if any_plotted:
        ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 7. Mean speedup (ALL_PASS only)
    ax = axes[2, 0]
    has_speedup = summary["n_speedup_measured"].sum() > 0
    if has_speedup:
        sp_means = pd.to_numeric(summary["mean_speedup"], errors="coerce").fillna(0).tolist()
        bars = ax.bar(x, sp_means, width=bar_w, color=palette, edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.axhline(1.0, color="grey", lw=0.8, ls="--", label="baseline (1x)")
        ax.set_ylim(0, max(sp_means) * 1.3 + 0.1 if max(sp_means) > 0 else 1.3)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No speedup data\n(re-run with --do-speedup)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="grey")
    ax.set_title("Mean Speedup\n(ALL_PASS samples only)", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("speedup vs unoptimized")
    ax.grid(axis="y", alpha=0.3)

    # 8. Paper average speedup
    ax = axes[2, 1]
    if "paper_avg_speedup" in summary.columns:
        avg_vals = pd.to_numeric(
            summary["paper_avg_speedup"], errors="coerce"
        ).fillna(1.0).tolist()
        bars = ax.bar(x, avg_vals, width=bar_w, color=palette,
                      edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fmt="%.3fx", padding=3, fontsize=9)
        ax.axhline(1.0, color="grey", lw=0.8, ls="--", label="baseline (1x)")
        ax.axhline(1.4, color="#d62728", lw=0.8, ls=":", label="paper ~1.4x")
        ax.set_ylim(0.95, max(max(avg_vals), 1.45) * 1.05)
        ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Average Speedup (paper metric)\nmean(max(1.0, speedup)) over all samples",
                 fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("average speedup")
    ax.grid(axis="y", alpha=0.3)

    # 9. All-sample speedup violin (failures at 0x)
    ax = axes[2, 2]
    if has_speedup:
        violin_data = []
        violin_labels = []
        violin_colors = []
        for model, color, label in zip(models, palette, labels):
            if model not in details:
                continue
            vals_sp = _all_sample_speedups(details[model])
            if len(vals_sp):
                violin_data.append(np.clip(vals_sp, 0, 2.0))
                violin_labels.append(label)
                violin_colors.append(color)

        if violin_data:
            parts = ax.violinplot(violin_data, showmeans=True, showmedians=True, widths=0.65)
            for body, color in zip(parts["bodies"], violin_colors):
                body.set_facecolor(color)
                body.set_edgecolor("white")
                body.set_alpha(0.65)
            for key in ("cbars", "cmins", "cmaxes", "cmeans", "cmedians"):
                if key in parts:
                    parts[key].set_color("black" if key in ("cmeans", "cmedians") else "grey")
                    parts[key].set_linewidth(1.0)
            ax.set_xticks(range(1, len(violin_data) + 1))
            ax.set_xticklabels(violin_labels, fontsize=9)
            ax.axhline(0, color="#d62728", lw=1.0, ls=":", label="failure = 0x")
            ax.axhline(1.0, color="grey", lw=0.8, ls="--", label="baseline (1x)")
            ax.set_ylim(-0.05, 2.0)
            ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "No speedup data\n(re-run with --do-speedup)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10, color="grey")
    ax.set_title("All-Sample Speedup Shape\n(raw speedup; failures at 0x, clipped at 2x)", fontweight="bold")
    ax.set_ylabel("speedup")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "infer_comparison.png")
    args = parser.parse_args()
    plot(args.out)


if __name__ == "__main__":
    main()

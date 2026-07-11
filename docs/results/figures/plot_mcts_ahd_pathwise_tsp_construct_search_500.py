"""Render the equal-budget MCTS-AHD and PathWise TSP Construct search curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


MAX_SAMPLES = 500
RUNS = {
    "MCTS-AHD": ("mcts_ahd", ("20260709_213505", "20260709_213507", "20260709_213510"), "#2A9D8F", "#A8DADC"),
    "PathWise": ("pathwise", ("20260710_123444", "20260710_123450", "20260710_123456"), "#E76F51", "#FFB4A2"),
}
RESULTS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "tsp_construct"
OUTPUT_STEM = RESULTS_DIR / "mcts-ahd-pathwise-qwen36-27b-tsp-construct-search-curve-500"


def _load_scores(method: str, run_name: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    for path in (EXPERIMENTS_DIR / method / run_name / "logs" / "samples").glob("samples_*.json"):
        if path.name == "samples_best.json":
            continue
        for record in json.loads(path.read_text(encoding="utf-8")):
            score = record.get("score")
            sample_order = record.get("sample_order")
            if isinstance(score, (int, float)) and np.isfinite(score) and sample_order is not None:
                scores[int(sample_order)] = float(score)
    return scores


def _best_so_far(scores: dict[int, float]) -> np.ndarray:
    curve = np.full(MAX_SAMPLES, np.nan, dtype=float)
    best = -np.inf
    for sample_order in range(1, MAX_SAMPLES + 1):
        if sample_order in scores:
            best = max(best, scores[sample_order])
        if np.isfinite(best):
            curve[sample_order - 1] = best
    return curve


def main() -> None:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 12, "axes.labelsize": 15, "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11, "figure.dpi": 300, "savefig.dpi": 300})
    fig, ax = plt.subplots(figsize=(6.7, 3.8))
    x = np.arange(1, MAX_SAMPLES + 1)
    handles = []
    for label, (method, run_names, color, band_color) in RUNS.items():
        curves = np.vstack([_best_so_far(_load_scores(method, run_name)) for run_name in run_names])
        mean = np.nanmean(curves, axis=0)
        lower = np.nanmin(curves, axis=0)
        upper = np.nanmax(curves, axis=0)
        ax.fill_between(x, lower, upper, step="post", color=band_color, alpha=0.45, linewidth=0)
        line, = ax.plot(x, mean, drawstyle="steps-post", color=color, linewidth=2.3, label=f"{label} mean", zorder=3)
        handles.extend([line, Patch(facecolor=band_color, edgecolor="none", alpha=0.45, label=f"{label} min-max")])
    ax.set_xlim(0, MAX_SAMPLES)
    ax.set_ylim(-7.0, -6.0)
    ax.set_xlabel("Number of Evaluations on D")
    ax.set_ylabel("Best Training Score on D (higher is better)")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.legend(handles=handles, loc="lower right", ncol=2, frameon=True, framealpha=0.95, edgecolor="#444444")
    fig.tight_layout()
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300)
    print(f"Wrote {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()

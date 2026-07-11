"""Render the three-run TraceAAD best-so-far training curve."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


RUN_NAMES = ("20260710_203531", "20260710_203541", "20260710_203551")
MAX_SAMPLES = 1000
RESULTS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = (
    Path(__file__).resolve().parents[3] / "experiments" / "tsp_construct" / "traceaad"
)
OUTPUT_STEM = RESULTS_DIR / "traceaad-qwen36-27b-tsp-construct-search-curve"


def _load_scores(run_name: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    samples_dir = EXPERIMENTS_DIR / run_name / "logs" / "samples"
    for path in samples_dir.glob("samples_*.json"):
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
    curves = np.vstack([_best_so_far(_load_scores(run_name)) for run_name in RUN_NAMES])
    x = np.arange(1, MAX_SAMPLES + 1)
    mean = np.nanmean(curves, axis=0)
    lower = np.nanmin(curves, axis=0)
    upper = np.nanmax(curves, axis=0)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 12,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    color = "#7B2CBF"
    band_color = "#CDB4DB"
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.fill_between(x, lower, upper, step="post", color=band_color, alpha=0.5, linewidth=0)
    line, = ax.plot(x, mean, drawstyle="steps-post", color=color, linewidth=2.3, label="TraceAAD mean", zorder=3)
    ax.set_xlim(0, MAX_SAMPLES)
    ax.set_ylim(-7.0, -5.8)
    ax.set_xlabel("Number of Evaluations on D")
    ax.set_ylabel("Best Training Score (higher is better)")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    band = Patch(facecolor=band_color, edgecolor="none", alpha=0.5, label="Min-max across 3 runs")
    ax.legend(handles=[line, band], loc="lower right", frameon=True, framealpha=0.95, edgecolor="#444444")
    fig.tight_layout()
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300)
    print(f"Wrote {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()

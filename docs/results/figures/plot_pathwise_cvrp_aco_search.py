"""Plot the three-run PathWise CVRP-ACO best-so-far search curve."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


RESULTS_DIR = Path(__file__).resolve().parent
EXPERIMENTS_DIR = Path(__file__).resolve().parents[3] / "experiments" / "cvrp_aco" / "pathwise"
RUN_NAMES = ("20260711_115024", "20260711_192005", "20260711_192010")
MAX_SAMPLES = 500
OUTPUT_STEM = RESULTS_DIR / "pathwise-qwen36-27b-cvrp-aco-search-curve"


def _load_scores(run_name: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    samples_dir = EXPERIMENTS_DIR / run_name / "logs" / "samples"
    for path in samples_dir.glob("samples_*.json"):
        if path.name == "samples_best.json":
            continue
        for record in json.loads(path.read_text(encoding="utf-8")):
            order = record.get("sample_order")
            score = record.get("score")
            if isinstance(order, int) and isinstance(score, (int, float)) and np.isfinite(score):
                scores[order] = float(score)
    return scores


def _best_so_far(scores: dict[int, float]) -> np.ndarray:
    curve = np.full(MAX_SAMPLES, np.nan, dtype=float)
    best = -np.inf
    for order in range(1, MAX_SAMPLES + 1):
        if order in scores:
            best = max(best, scores[order])
        if np.isfinite(best):
            curve[order - 1] = best
    return curve


def main() -> None:
    for run_name in RUN_NAMES:
        summary_path = EXPERIMENTS_DIR / run_name / "logs" / "run_summary.json"
        if not summary_path.exists() or json.loads(summary_path.read_text(encoding="utf-8")).get("status") != "finished":
            raise RuntimeError(f"Run is not finished: {run_name}")
    curves = np.vstack([_best_so_far(_load_scores(name)) for name in RUN_NAMES])
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
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.fill_between(x, lower, upper, step="post", color="#FFB4A2", alpha=0.5, linewidth=0)
    line, = ax.plot(x, mean, drawstyle="steps-post", color="#E76F51", linewidth=2.3, label="PathWise mean")
    ax.set_xlim(0, MAX_SAMPLES)
    ax.set_xlabel("Number of Evaluations on D")
    ax.set_ylabel("Best Training Score on D (higher is better)")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    band = Patch(facecolor="#FFB4A2", edgecolor="none", alpha=0.5, label="Min-max across 3 runs")
    ax.legend(handles=[line, band], loc="lower right", frameon=True, framealpha=0.95, edgecolor="#444444")
    fig.tight_layout()
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300)
    print(f"Wrote {OUTPUT_STEM.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()

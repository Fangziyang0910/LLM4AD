"""Plot PathWise best-so-far search curves for TSP-GLS (batch 20260720_140109)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "docs" / "results" / "tsp_gls"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "tsp_gls"
METHOD = {
    "label": "PathWise",
    "directory": "pathwise",
    "runs": (
        "20260720_140109_tspgls_rep1",
        "20260720_140109_tspgls_rep2",
        "20260720_140109_tspgls_rep3",
    ),
    "budget": 500,
    "color": "#E76F51",
    "band": "#FFB4A2",
}


def _load_scores(run_name: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    samples_dir = EXPERIMENTS_DIR / METHOD["directory"] / run_name / "logs" / "samples"
    for path in samples_dir.glob("samples_*.json"):
        if path.name == "samples_best.json":
            continue
        for record in json.loads(path.read_text(encoding="utf-8")):
            score = record.get("score")
            sample_order = record.get("sample_order")
            if isinstance(score, (int, float)) and np.isfinite(score) and sample_order is not None:
                scores[int(sample_order)] = float(score)
    return scores


def _best_so_far(scores: dict[int, float], budget: int) -> np.ndarray:
    curve = np.full(budget, np.nan, dtype=float)
    best = -np.inf
    for sample_order in range(1, budget + 1):
        if sample_order in scores:
            best = max(best, scores[sample_order])
        if np.isfinite(best):
            curve[sample_order - 1] = best
    return curve


def _style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"],
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def main() -> None:
    budget = int(METHOD["budget"])
    curves = np.vstack([_best_so_far(_load_scores(run_name), budget) for run_name in METHOD["runs"]])
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(1, budget + 1)
    mean = np.nanmean(curves, axis=0)
    lower = np.nanmin(curves, axis=0)
    upper = np.nanmax(curves, axis=0)
    ax.fill_between(x, lower, upper, step="post", color=METHOD["band"], alpha=0.25, linewidth=0)
    line, = ax.plot(
        x,
        mean,
        drawstyle="steps-post",
        color=METHOD["color"],
        linewidth=2.2,
        label=f"{METHOD['label']} mean ({budget} evaluations)",
        zorder=3,
    )
    values = curves[np.isfinite(curves)]
    y_min = float(np.percentile(values, 5))
    padding = max((float(values.max()) - y_min) * 0.08, 0.001)
    ax.set_xlim(0, budget)
    ax.set_ylim(y_min - padding * 0.2, float(values.max()) + padding)
    handles = [
        line,
        Patch(facecolor="#999999", edgecolor="none", alpha=0.25, label="Min-max range across three runs"),
    ]
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best-so-far score (higher is better)")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95, edgecolor="#444444")
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "搜索曲线_PathWise.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

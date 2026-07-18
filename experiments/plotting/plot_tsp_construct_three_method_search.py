"""Compare MCTS-AHD, PathWise, and TraceAAD best-so-far training curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "docs" / "results" / "tsp_construct"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "tsp_construct"
OUTPUT_STEM = RESULTS_DIR / "搜索曲线"
METHODS = {
    "MCTS-AHD": {
        "directory": "mcts_ahd",
        "runs": ("20260709_213505", "20260709_213507", "20260709_213510"),
        "budget": 1000,
        "color": "#247BA0",
        "band": "#A9D6E5",
    },
    "PathWise": {
        "directory": "pathwise",
        "runs": ("20260710_123444", "20260710_123450", "20260710_123456"),
        "budget": 500,
        "color": "#E76F51",
        "band": "#FFB4A2",
    },
    "TraceAAD version1": {
        "directory": "traceaad/version1",
        "runs": ("20260710_203531", "20260710_203541", "20260710_203551"),
        "budget": 1000,
        "color": "#2A9D5B",
        "band": "#A8D5BA",
    },
}


def _load_scores(method: str, run_name: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    samples_dir = EXPERIMENTS_DIR / method / run_name / "logs" / "samples"
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


def main() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"],
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    handles = []
    for label, config in METHODS.items():
        budget = config["budget"]
        curves = np.vstack(
            [
                _best_so_far(_load_scores(config["directory"], run_name), budget)
                for run_name in config["runs"]
            ]
        )
        x = np.arange(1, budget + 1)
        mean = np.nanmean(curves, axis=0)
        lower = np.nanmin(curves, axis=0)
        upper = np.nanmax(curves, axis=0)
        ax.fill_between(x, lower, upper, step="post", color=config["band"], alpha=0.25, linewidth=0)
        line, = ax.plot(
            x,
            mean,
            drawstyle="steps-post",
            color=config["color"],
            linewidth=2.2,
            label=f"{label} 平均（{budget} 次评估）",
            zorder=3,
        )
        handles.append(line)
    handles.append(Patch(facecolor="#999999", edgecolor="none", alpha=0.25, label="三次运行的最小-最大范围"))
    ax.set_xlim(0, 1000)
    ax.set_ylim(-7.0, -5.8)
    ax.set_xlabel("评估次数")
    ax.set_ylabel("训练集最佳分数（越高越好）")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95, edgecolor="#444444")
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300)
    print(f"Wrote {OUTPUT_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()

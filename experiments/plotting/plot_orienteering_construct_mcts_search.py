"""Render the three-run MCTS-AHD search curve for OP."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "docs" / "results"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "orienteering_construct" / "mcts_ahd"
RUNS = ("20260713_125413", "20260713_125707", "20260713_125712")
BUDGET = 1000
OUTPUT_PATH = RESULTS_DIR / "mcts-ahd-qwen36-27b-orienteering-construct-search-curve.png"


def _load_scores(run_name: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    samples_dir = EXPERIMENTS_DIR / run_name / "logs" / "samples"
    for path in samples_dir.glob("samples_*.json"):
        if path.name == "samples_best.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for record in data:
            order = record.get("sample_order")
            score = record.get("score")
            if isinstance(order, int) and isinstance(score, (int, float)) and np.isfinite(score):
                scores[order] = float(score)
    return scores


def _best_so_far(scores: dict[int, float]) -> np.ndarray:
    curve = np.full(BUDGET, np.nan, dtype=float)
    best = -np.inf
    for order in range(1, BUDGET + 1):
        if order in scores:
            best = max(best, scores[order])
        if np.isfinite(best):
            curve[order - 1] = best
    return curve


def main() -> None:
    curves = np.vstack([_best_so_far(_load_scores(run_name)) for run_name in RUNS])
    if np.isnan(curves).any():
        raise RuntimeError("one or more completed OP runs have missing sample scores")

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
    x = np.arange(1, BUDGET + 1)
    mean = curves.mean(axis=0)
    lower = curves.min(axis=0)
    upper = curves.max(axis=0)
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.fill_between(x, lower, upper, step="post", color="#A9D6E5", alpha=0.35, linewidth=0)
    ax.plot(x, mean, drawstyle="steps-post", color="#247BA0", linewidth=2.2, label="MCTS-AHD 平均")
    ax.set_xlim(0, BUDGET)
    ax.set_xlabel("评估次数")
    ax.set_ylabel("训练集最佳分数（越高越好）")
    ax.set_title("Orienteering Construct 搜索曲线")
    ax.grid(True, color="#D9D9D9", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

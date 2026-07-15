"""Render OP search curves for completed methods in one figure."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "docs" / "results" / "orienteering_construct"
OUTPUT_PATH = RESULTS_DIR / "搜索曲线_方法对比.png"


@dataclass(frozen=True)
class MethodSpec:
    label: str
    method_dir: Path
    runs: tuple[str, ...]
    budget: int
    color: str
    band_color: str


METHODS = (
    MethodSpec(
        label="MCTS-AHD",
        method_dir=PROJECT_ROOT / "experiments" / "orienteering_construct" / "mcts_ahd",
        runs=("20260713_125413", "20260713_125707", "20260713_125712"),
        budget=1000,
        color="#0072B2",
        band_color="#56B4E9",
    ),
    MethodSpec(
        label="PathWise",
        method_dir=PROJECT_ROOT / "experiments" / "orienteering_construct" / "pathwise",
        runs=("20260714_105543_rep1", "20260714_105543_rep2", "20260714_105543_rep3"),
        budget=500,
        color="#D55E00",
        band_color="#E69F00",
    ),
    MethodSpec(
        label="TraceAAD",
        method_dir=PROJECT_ROOT / "experiments" / "orienteering_construct" / "traceaad",
        runs=("20260714_141500_rep1", "20260714_141505_rep2", "20260714_141510_rep3"),
        budget=1000,
        color="#009E73",
        band_color="#66C2A5",
    ),
)


def _load_scores(method: MethodSpec, run_name: str) -> dict[int, float]:
    summary_path = method.method_dir / run_name / "logs" / "run_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Run is not finished: {method.label} {run_name}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "finished" or summary.get("search_aborted"):
        raise RuntimeError(f"Run is not complete: {method.label} {run_name}")

    scores: dict[int, float] = {}
    samples_dir = method.method_dir / run_name / "logs" / "samples"
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


def _best_so_far(scores: dict[int, float], budget: int) -> np.ndarray:
    curve = np.full(budget, np.nan, dtype=float)
    best = -np.inf
    for order in range(1, budget + 1):
        if order in scores:
            best = max(best, scores[order])
        if np.isfinite(best):
            curve[order - 1] = best
    return curve


def _method_curves(method: MethodSpec) -> np.ndarray:
    curves = np.vstack(
        [_best_so_far(_load_scores(method, run_name), method.budget) for run_name in method.runs]
    )
    if np.isnan(curves).any():
        raise RuntimeError(f"one or more completed runs have missing sample scores: {method.label}")
    return curves


def main() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"],
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for method in METHODS:
        curves = _method_curves(method)
        x = np.arange(1, method.budget + 1)
        mean = curves.mean(axis=0)
        lower = curves.min(axis=0)
        upper = curves.max(axis=0)
        ax.fill_between(x, lower, upper, step="post", color=method.band_color, alpha=0.22, linewidth=0)
        ax.plot(
            x,
            mean,
            drawstyle="steps-post",
            color=method.color,
            linewidth=2.2,
            label=f"{method.label} 平均",
        )

    ax.set_xlim(0, max(method.budget for method in METHODS))
    ax.set_ylim(bottom=13.0)
    ax.set_xlabel("评估次数")
    ax.set_ylabel("训练集最佳分数（越高越好）")
    ax.set_title("Orienteering Construct 方法搜索曲线")
    ax.grid(True, color="#D9D9D9", linewidth=0.8)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

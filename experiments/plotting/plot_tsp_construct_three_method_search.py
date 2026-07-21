"""Compare MCTS-AHD, PathWise, and TraceAAD search curves for TSP Construct."""

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
Y_MIN = -8.0
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
    "TraceAAD version2": {
        "directory": "traceaad/version2",
        "runs": ("20260718_174552_tsp_rep1", "20260718_174552_tsp_rep2", "20260718_174552_tsp_rep3"),
        "budget": 1000,
        "color": "#6A4C93",
        "band": "#C9B1FF",
    },
}


def _load_scores(directory: str, run_name: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    samples_dir = EXPERIMENTS_DIR / directory / run_name / "logs" / "samples"
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


def _method_curves(label: str) -> np.ndarray:
    config = METHODS[label]
    return np.vstack(
        [
            _best_so_far(_load_scores(config["directory"], run_name), config["budget"])
            for run_name in config["runs"]
        ]
    )


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


def _render(method_names: tuple[str, ...], output_stem: str) -> None:
    data = [(name, _method_curves(name)) for name in method_names]
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    handles = []
    all_curves = []
    max_budget = max(METHODS[name]["budget"] for name in method_names)
    for name, curves in data:
        config = METHODS[name]
        budget = config["budget"]
        x = np.arange(1, budget + 1)
        mean = np.nanmean(curves, axis=0)
        lower = np.nanmin(curves, axis=0)
        upper = np.nanmax(curves, axis=0)
        all_curves.append(curves)
        ax.fill_between(x, lower, upper, step="post", color=config["band"], alpha=0.25, linewidth=0)
        line, = ax.plot(
            x,
            mean,
            drawstyle="steps-post",
            color=config["color"],
            linewidth=2.2,
            label=f"{name} mean ({budget} evaluations)",
            zorder=3,
        )
        handles.append(line)
    values = np.concatenate([curve[np.isfinite(curve)] for curves in all_curves for curve in curves])
    visible_values = values[values >= Y_MIN]
    padding = max((float(visible_values.max()) - Y_MIN) * 0.08, 0.05)
    ax.set_xlim(0, max_budget)
    ax.set_ylim(Y_MIN, float(visible_values.max()) + padding)
    handles.append(Patch(facecolor="#999999", edgecolor="none", alpha=0.25, label="Min-max range across three runs"))
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Best-so-far score (higher is better)")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.legend(handles=handles, loc="lower right", frameon=True, framealpha=0.95, edgecolor="#444444")
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / output_stem
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Wrote {output.with_suffix('.png')}")


def main() -> None:
    _render(tuple(METHODS), "搜索曲线")


if __name__ == "__main__":
    main()

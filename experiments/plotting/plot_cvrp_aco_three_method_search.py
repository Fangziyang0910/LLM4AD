"""Render CVRP-ACO search curves for completed runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "docs" / "results" / "cvrp_aco"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "cvrp_aco"
METHODS = {
    "MCTS-AHD": {
        "directory": "mcts_ahd",
        "runs": ("20260711_115024", "20260712_021911", "20260712_021957"),
        "budget": 1000,
        "color": "#247BA0",
        "band": "#A9D6E5",
    },
    "PathWise": {
        "directory": "pathwise",
        "runs": ("20260711_115024", "20260711_192005", "20260711_192010"),
        "budget": 500,
        "color": "#E76F51",
        "band": "#FFB4A2",
    },
    "TraceAAD version2": {
        "directory": "traceaad/version2",
        "runs": (
            "20260718_193305_cvrp_rep1",
            "20260718_193305_cvrp_rep2",
            "20260718_193305_cvrp_rep3",
        ),
        "budget": 1000,
        "color": "#6A4C93",
        "band": "#C9B1FF",
    },
    "TraceAAD version3": {
        "directory": "traceaad/version3",
        "runs": (
            "20260720_233339_cvrp_v3_rep1",
            "20260720_233339_cvrp_v3_rep2",
            "20260720_233339_cvrp_v3_rep3",
        ),
        "budget": 1000,
        "color": "#2A9D5B",
        "band": "#A8D5BA",
    },
    "TraceAAD version4": {
        "directory": "traceaad/version4",
        "runs": (
            "20260723_204526_cvrp_v4_rep1",
            "20260723_204526_cvrp_v4_rep2",
            "20260723_204526_cvrp_v4_rep3",
        ),
        "budget": 1000,
        "color": "#F4A261",
        "band": "#FAD7A0",
    },
    "TraceAAD version5": {
        "directory": "traceaad_v5/version5",
        "runs": (
            "20260724_220827_cvrp_v5_rep1",
            "20260724_220827_cvrp_v5_rep2",
            "20260724_220827_cvrp_v5_rep3",
        ),
        "budget": 1000,
        "color": "#D55E00",
        "band": "#F3B49F",
    },
}
Y_MIN = -13.0


def _load_scores(directory: str, run_name: str) -> dict[int, float]:
    scores: dict[int, float] = {}
    samples_dir = EXPERIMENTS_DIR / directory / run_name / "logs" / "samples"
    for path in samples_dir.glob("samples_*.json"):
        if path.name == "samples_best.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue
        for record in data:
            score = record.get("score")
            order = record.get("sample_order")
            if (
                isinstance(score, (int, float))
                and np.isfinite(score)
                and order is not None
            ):
                scores[int(order)] = float(score)
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


def _method_curves(label: str) -> np.ndarray:
    config = METHODS[label]
    curves = []
    for run_name in config["runs"]:
        summary_path = (
            EXPERIMENTS_DIR
            / config["directory"]
            / run_name
            / "logs"
            / "run_summary.json"
        )
        if not summary_path.exists():
            raise RuntimeError(f"Run is not finished: {run_name}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "finished":
            raise RuntimeError(
                f"Run is not finished: {run_name} status={summary.get('status')!r}"
            )
        curves.append(
            _best_so_far(_load_scores(config["directory"], run_name), config["budget"])
        )
    return np.vstack(curves)


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


def _limits(
    curves: list[np.ndarray], *, floor: float | None = Y_MIN
) -> tuple[float, float]:
    values = np.concatenate([curve[np.isfinite(curve)] for curve in curves])
    high = float(values.max())
    low = float(values.min()) if floor is None else floor
    if floor is None:
        padding = max((high - low) * 0.08, 0.2)
        return low - padding, high + padding
    padding = max((high - low) * 0.08, 0.2)
    return low, high + padding


def _render(
    method_names: tuple[str, ...], output_stem: str, *, y_floor: float | None = Y_MIN
) -> None:
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
        ax.fill_between(
            x, lower, upper, step="post", color=config["band"], alpha=0.25, linewidth=0
        )
        (line,) = ax.plot(
            x,
            mean,
            drawstyle="steps-post",
            color=config["color"],
            linewidth=2.2,
            label=f"{name} 平均（{budget} 次评估）",
            zorder=3,
        )
        handles.append(line)
    flattened = [curve for method_curves in all_curves for curve in method_curves]
    handles.append(
        Patch(
            facecolor="#999999",
            edgecolor="none",
            alpha=0.25,
            label="已完成运行的最小-最大范围",
        )
    )
    ax.set_xlim(0, max_budget)
    ax.set_ylim(*_limits(flattened, floor=y_floor))
    ax.set_xlabel("评估次数")
    ax.set_ylabel("训练集最佳分数（越高越好）")
    ax.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.legend(
        handles=handles,
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#444444",
    )
    fig.tight_layout()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / output_stem
    fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)
    print(f"Wrote {output.with_suffix('.png')}")


def main() -> None:
    _render(tuple(METHODS), "搜索曲线", y_floor=Y_MIN)


if __name__ == "__main__":
    main()

"""Plot TSP Construct search curves for MCTS-AHD, PathWise, TraceAAD v4 / v5.1 / v5.2 / v5.3."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "experiments" / "tsp_construct"
RESULTS_DIR = ROOT / "docs" / "results" / "tsp_construct"
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
    "TraceAAD v4": {
        "directory": "traceaad/version4",
        "runs": (
            "20260723_181743_tspc_v4_rep1",
            "20260723_181743_tspc_v4_rep2",
            "20260723_181743_tspc_v4_rep3",
        ),
        "budget": 1000,
        "color": "#F4A261",
        "band": "#FAD7A0",
    },
    "TraceAAD v5.1": {
        "directory": "traceaad_v5/version5_1",
        "runs": (
            "20260725_234854_tspc_v51_rep1",
            "20260725_234854_tspc_v51_rep2",
            "20260725_234854_tspc_v51_rep3",
        ),
        "budget": 1000,
        "color": "#0072B2",
        "band": "#9EC9E2",
    },
    "TraceAAD v5.2": {
        "directory": "traceaad_v5/version5_2",
        "runs": (
            "20260727_210010_tspc_v52_rep1",
            "20260727_210010_tspc_v52_rep2",
            "20260727_210010_tspc_v52_rep3",
        ),
        "budget": 1000,
        "color": "#009E73",
        "band": "#7DCEA0",
    },
    "TraceAAD v5.3": {
        "directory": "traceaad_v5/version5_3",
        "runs": (
            "20260728_151736_tspc_v53_rep1",
            "20260728_151736_tspc_v53_rep2",
            "20260728_151736_tspc_v53_rep3",
        ),
        "budget": 1000,
        "color": "#CC79A7",
        "band": "#E6B8D0",
    },
    "TraceAAD v5.4": {
        "directory": "traceaad_v5/version5_4",
        "runs": (
            "20260729_093300_tspc_v54_rep1",
            "20260729_093300_tspc_v54_rep2",
            "20260729_093300_tspc_v54_rep3",
        ),
        "budget": 1000,
        "color": "#E69F00",
        "band": "#F6D58A",
    },
}


def _load_curve(directory: str, run_name: str, budget: int) -> np.ndarray:
    run_dir = EXPERIMENTS_DIR / directory / run_name
    summary = json.loads((run_dir / "logs" / "run_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "finished" or summary.get("search_aborted"):
        raise RuntimeError(f"Run is not complete: {run_name}")

    scores = np.full(budget, -np.inf)
    for path in (run_dir / "logs" / "samples").glob("samples_*.json"):
        if path.name == "samples_best.json":
            continue
        for sample in json.loads(path.read_text(encoding="utf-8")):
            order, score = sample.get("sample_order"), sample.get("score")
            if isinstance(order, int) and isinstance(score, (int, float)) and 1 <= order <= budget:
                scores[order - 1] = max(scores[order - 1], float(score))
    curve = np.maximum.accumulate(scores)
    curve[~np.isfinite(curve)] = np.nan
    return curve


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(6.75, 3.6))
    handles = []
    all_values = []
    max_budget = max(config["budget"] for config in METHODS.values())
    for name, config in METHODS.items():
        budget = config["budget"]
        curves = np.vstack(
            [_load_curve(config["directory"], run, budget) for run in config["runs"]]
        )
        x = np.arange(1, budget + 1)
        all_values.append(curves[np.isfinite(curves)])
        ax.fill_between(
            x,
            np.nanmin(curves, axis=0),
            np.nanmax(curves, axis=0),
            step="post",
            color=config["band"],
            alpha=0.3,
            linewidth=0,
        )
        (line,) = ax.plot(
            x,
            np.nanmean(curves, axis=0),
            drawstyle="steps-post",
            color=config["color"],
            linewidth=2.2,
            label=f"{name} mean ({budget:,})",
        )
        handles.append(line)

    handles.append(
        Patch(
            facecolor="#999999",
            edgecolor="none",
            alpha=0.3,
            label="Min–max range across three runs",
        )
    )
    values = np.concatenate(all_values)
    y_min = float(np.percentile(values, 5))
    padding = max((float(values.max()) - y_min) * 0.08, 0.2)
    ax.set(
        xlim=(0, max_budget),
        ylim=(y_min - padding * 0.2, float(values.max()) + padding),
        xlabel="Evaluations",
        ylabel="Best-so-far score (higher is better)",
    )
    ax.grid(alpha=0.2)
    ax.legend(handles=handles, frameon=False, loc="lower right")
    fig.tight_layout()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / "搜索曲线.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

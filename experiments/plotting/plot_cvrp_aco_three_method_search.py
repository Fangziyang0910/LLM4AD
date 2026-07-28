"""Plot CVRP-ACO search curves for MCTS-AHD, PathWise, TraceAAD v4 / v5.1 / v5.2."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "experiments" / "cvrp_aco"
RESULTS_DIR = ROOT / "docs" / "results" / "cvrp_aco"
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
    "TraceAAD v4": {
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
    "TraceAAD v5.1": {
        "directory": "traceaad_v5/version5_1",
        "runs": (
            "20260725_234854_cvrp_v51_rep1",
            "20260725_234854_cvrp_v51_rep2",
            "20260725_234854_cvrp_v51_rep3",
        ),
        "budget": 1000,
        "color": "#0072B2",
        "band": "#9EC9E2",
    },
    "TraceAAD v5.2": {
        "directory": "traceaad_v5/version5_2",
        "runs": (
            "20260727_210010_cvrp_v52_rep1",
            "20260727_210010_cvrp_v52_rep2",
            "20260727_210010_cvrp_v52_rep3",
        ),
        "budget": 1000,
        "color": "#009E73",
        "band": "#7DCEA0",
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
        mean = np.nanmean(curves, axis=0)
        lower = np.nanmin(curves, axis=0)
        upper = np.nanmax(curves, axis=0)
        all_values.append(curves[np.isfinite(curves)])
        ax.fill_between(
            x,
            lower,
            upper,
            step="post",
            color=config["band"],
            alpha=0.3,
            linewidth=0,
        )
        (line,) = ax.plot(
            x,
            mean,
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
    # Touch a fresh write so markdown previews pick up the new asset.
    output.touch()
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

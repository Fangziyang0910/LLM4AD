"""Plot TraceAAD v4/v5 OP-ACO training curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "experiments" / "op_aco"
RESULTS_DIR = ROOT / "docs" / "results" / "op_aco"
BUDGET = 1000
METHODS = {
    "TraceAAD v4": {
        "directory": "traceaad/version4",
        "runs": (
            "20260723_204526_opaco_v4_rep1",
            "20260723_204526_opaco_v4_rep2",
            "20260723_204526_opaco_v4_rep3",
        ),
        "color": "#F4A261",
        "band": "#FAD7A0",
    },
    "TraceAAD v5": {
        "directory": "traceaad_v5/version5",
        "runs": (
            "20260724_220827_opaco_v5_rep1",
            "20260724_220827_opaco_v5_rep2",
            "20260724_220827_opaco_v5_rep3",
        ),
        "color": "#D55E00",
        "band": "#F3B49F",
    },
}


def _load_curve(directory: str, run_name: str) -> np.ndarray:
    run_dir = EXPERIMENTS_DIR / directory / run_name
    summary = json.loads((run_dir / "logs" / "run_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "finished" or summary.get("search_aborted"):
        raise RuntimeError(f"Run is not complete: {run_name}")

    scores = np.full(BUDGET, -np.inf)
    for path in (run_dir / "logs" / "samples").glob("samples_*.json"):
        if path.name == "samples_best.json":
            continue
        for sample in json.loads(path.read_text(encoding="utf-8")):
            order, score = sample.get("sample_order"), sample.get("score")
            if isinstance(order, int) and isinstance(score, (int, float)):
                scores[order - 1] = max(scores[order - 1], float(score))
    return np.maximum.accumulate(scores)


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
    x = np.arange(1, BUDGET + 1)
    for name, config in METHODS.items():
        curves = np.vstack(
            [_load_curve(config["directory"], run) for run in config["runs"]]
        )
        ax.fill_between(
            x,
            curves.min(axis=0),
            curves.max(axis=0),
            step="post",
            color=config["band"],
            alpha=0.3,
            linewidth=0,
        )
        (line,) = ax.plot(
            x,
            curves.mean(axis=0),
            drawstyle="steps-post",
            color=config["color"],
            linewidth=2.2,
            label=f"{name} mean",
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
    ax.set(xlim=(0, BUDGET), xlabel="Evaluations", ylabel="Best-so-far score (higher is better)")
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

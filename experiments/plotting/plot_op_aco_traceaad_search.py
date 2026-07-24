"""Plot the TraceAAD v4 OP-ACO training curve."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "experiments" / "op_aco" / "traceaad" / "version4"
RUNS = [f"20260723_204526_opaco_v4_rep{i}" for i in range(1, 4)]
BUDGET = 1000


def load_curve(run_name: str) -> np.ndarray:
    run_dir = RUN_ROOT / run_name
    summary = json.loads((run_dir / "logs" / "run_summary.json").read_text())
    if summary.get("status") != "finished" or summary.get("search_aborted"):
        raise RuntimeError(f"Run is not complete: {run_name}")

    scores = np.full(BUDGET, -np.inf)
    for path in (run_dir / "logs" / "samples").glob("samples_*.json"):
        if path.name == "samples_best.json":
            continue
        for sample in json.loads(path.read_text()):
            order, score = sample.get("sample_order"), sample.get("score")
            if isinstance(order, int) and isinstance(score, (int, float)):
                scores[order - 1] = max(scores[order - 1], float(score))
    return np.maximum.accumulate(scores)


def main() -> None:
    curves = np.vstack([load_curve(run) for run in RUNS])
    x = np.arange(1, BUDGET + 1)
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
    ax.fill_between(
        x,
        curves.min(axis=0),
        curves.max(axis=0),
        step="post",
        color="#F3B49F",
        alpha=0.3,
        label="Min–max range across three runs",
    )
    ax.plot(
        x,
        curves.mean(axis=0),
        drawstyle="steps-post",
        color="#D55E00",
        linewidth=2.2,
        label="TraceAAD v4 mean",
    )
    ax.set(xlim=(0, BUDGET), xlabel="Evaluations", ylabel="Best-so-far score (higher is better)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()

    output = ROOT / "docs" / "results" / "op_aco" / "搜索曲线.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

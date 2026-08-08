"""Plot per-task search curves for the 6-method comparison figure.

Methods: TraceAAD V9 (ours) + MCTS-AHD, PathWise, EoH, ReEvo, CALM.
Each task gets one figure with best-so-far curves (mean + min-max band over 3 runs).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]

TASKS = ["tsp_construct", "cvrp_aco", "op_aco", "online_bin_packing"]

METHODS = {
    "TraceAAD V9": {
        "color": "#D62728",
        "band": "#F5A9A9",
    },
    "MCTS-AHD": {
        "color": "#247BA0",
        "band": "#A9D6E5",
    },
    "PathWise": {
        "color": "#E76F51",
        "band": "#FFB4A2",
    },
    "EoH": {
        "color": "#CC79A7",
        "band": "#E8B9D5",
    },
    "ReEvo": {
        "color": "#56B4E9",
        "band": "#A8D8F0",
    },
    "CALM": {
        "color": "#8C8C8C",
        "band": "#C9C9C9",
    },
}

BUDGET = 1000


RUN_GLOBS = {
    "tsp_construct": {
        "TraceAAD V9": ["traceaad_v9/version9/*_v9_*_rep*"],
        "MCTS-AHD": ["mcts_ahd/20260709_2135*"],
        "PathWise": ["pathwise/20260730_1755_*_pw_rep*"],
        "EoH": ["eoh/eoh_paper_*_eoh_rep*"],
        "ReEvo": ["reevo/*_reevo_rep*"],
        "CALM": ["calm/*_calm_rep*"],
    },
    "cvrp_aco": {
        "TraceAAD V9": ["traceaad_v9/version9/*_v9_*_rep*"],
        "MCTS-AHD": [
            "mcts_ahd/20260711_115024",
            "mcts_ahd/20260712_021911",
            "mcts_ahd/20260712_021957",
        ],
        "PathWise": ["pathwise/20260730_1755_*_pw_rep*"],
        "EoH": ["eoh/eoh_paper_*_eoh_rep*"],
        "ReEvo": ["reevo/*_reevo_rep*"],
        "CALM": ["calm/*_calm_rep*"],
    },
    "op_aco": {
        "TraceAAD V9": ["traceaad_v9/version9/*_v9_*_rep*"],
        "MCTS-AHD": ["mcts_ahd/*mctsahd_rep*"],
        "PathWise": ["pathwise/20260730_1755_*_pw_rep*"],
        "EoH": ["eoh/eoh_paper_*_eoh_rep*"],
        "ReEvo": ["reevo/*_reevo_rep*"],
        "CALM": ["calm/*_calm_rep*"],
    },
    "online_bin_packing": {
        "TraceAAD V9": ["traceaad_v9/version9/*_v9_*_rep*"],
        "MCTS-AHD": ["mcts_ahd/*mctsahd_rep*"],
        "PathWise": ["pathwise/20260730_1755_*_pw_rep*"],
        "EoH": ["eoh/eoh_paper_*_eoh_rep*"],
        "ReEvo": ["reevo/*_reevo_rep*"],
        "CALM": ["calm/*_calm_rep*"],
    },
}


def run_dirs(task: str, method: str) -> list[Path]:
    td = ROOT / "experiments" / task
    dirs: list[Path] = []
    for pattern in RUN_GLOBS[task][method]:
        dirs.extend(td.glob(pattern))
    return sorted(dirs)


def _v9_points(run: Path) -> list[tuple[int, float]]:
    rows = [json.loads(l) for l in (run / "artifacts" / "candidates.jsonl").read_text().splitlines()]
    pts = []
    for r in rows:
        s = r.get("score")
        if r.get("status") == "ok" and isinstance(r.get("sample_order"), int) and isinstance(s, (int, float)):
            pts.append((r["sample_order"], float(s)))
    return pts


def _calm_points(run: Path) -> list[tuple[int, float]]:
    log = run / "logs" / "calm" / "output.log"
    pts: list[tuple[int, float]] = []
    eval_re = re.compile(r"Evals: (\d+)/\d+")
    perf_re = re.compile(r"Perf: (-?[\d.]+)")
    for line in log.read_text().splitlines():
        m = eval_re.search(line)
        p = perf_re.search(line)
        if m and p:
            pts.append((int(m.group(1)), float(p.group(1))))
    if not pts:
        return pts
    seed = None
    for f in sorted((run / "logs" / "samples").glob("samples_*.json")):
        for s in json.loads(f.read_text()):
            if isinstance(s.get("sample_order"), int) and isinstance(s.get("score"), (int, float)):
                if seed is None or s["sample_order"] < seed[0]:
                    seed = (s["sample_order"], float(s["score"]))
    if seed is not None:
        pts.append((seed[0], seed[1]))
    pts.sort(key=lambda x: x[0])
    return pts


def _generic_points(run: Path) -> list[tuple[int, float]]:
    pts: list[tuple[int, float]] = []
    for f in sorted((run / "logs" / "samples").glob("samples_*.json")):
        if f.name == "samples_best.json":
            continue
        for s in json.loads(f.read_text()):
            order, score = s.get("sample_order"), s.get("score")
            if isinstance(order, int) and isinstance(score, (int, float)):
                pts.append((order, float(score)))
    return pts


def load_curve(run: Path, method: str) -> np.ndarray:
    if method == "TraceAAD V9":
        pts = _v9_points(run)
    elif method == "CALM":
        pts = _calm_points(run)
    else:
        pts = _generic_points(run)
    arr = np.full(BUDGET, -np.inf)
    for n, s in pts:
        if 1 <= n <= BUDGET:
            arr[n - 1] = max(arr[n - 1], s)
    curve = np.maximum.accumulate(arr)
    finite = np.isfinite(curve)
    if finite.any():
        idx = np.where(finite)[0]
        curve[: idx[0]] = curve[idx[0]]
        curve[idx[-1] + 1 :] = curve[idx[-1]]
    return curve


def plot_task(task: str) -> Path:
    out = ROOT / "docs" / "experiments" / task
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.75, 3.95))
    handles = []
    all_values = []
    for name, cfg in METHODS.items():
        runs = [d for d in run_dirs(task, name) if (d / "logs").exists()]
        curves = np.vstack([load_curve(r, name) for r in runs])
        x = np.arange(1, BUDGET + 1)
        all_values.append(curves[np.isfinite(curves)])
        ax.fill_between(
            x,
            np.nanmin(curves, axis=0),
            np.nanmax(curves, axis=0),
            step="post",
            color=cfg["band"],
            alpha=0.3,
            linewidth=0,
        )
        (line,) = ax.plot(
            x,
            np.nanmean(curves, axis=0),
            drawstyle="steps-post",
            color=cfg["color"],
            linewidth=2.0,
            label=name,
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
        xlim=(0, BUDGET),
        ylim=(y_min - padding * 0.2, float(values.max()) + padding),
        xlabel="Evaluations",
        ylabel="Best-so-far score (higher is better)",
    )
    ax.grid(alpha=0.2)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.97, bottom=0.24)
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        fontsize=8.0,
        handlelength=2.6,
        handletextpad=0.6,
        columnspacing=1.3,
        labelspacing=0.45,
    )
    png = out / "搜索曲线.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(out / "搜索曲线.pdf", bbox_inches="tight")
    plt.close(fig)
    return png


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
    for task in TASKS:
        print("wrote", plot_task(task))


if __name__ == "__main__":
    main()

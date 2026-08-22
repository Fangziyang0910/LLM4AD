"""Plot per-task search curves for the TraceAAD version comparison figure.

Methods: TraceAAD V9.7 / V9.14 / V9.15. Each task gets one figure with
best-so-far curves (mean + min-max band over 3 runs). V9.7 reads
``artifacts/candidates.jsonl``; V9.14 / V9.15 read ``evaluations.csv``.
Every earlier TraceAAD version and the five external baselines lost their
per-sample histories in the 2026-08-21 cleanup; those curves survive only
in the git history of the committed figures. V9.16 joins the figure once
its batch finishes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]

TASKS = ["tsp_construct", "cvrp_aco", "op_aco", "online_bin_packing"]

METHODS = {
    "TraceAAD V9.7": {
        "color": "#FF7F0E",
        "band": "#FCC07E",
    },
    "TraceAAD V9.14": {
        "color": "#2CA02C",
        "band": "#A8D8A8",
    },
    "TraceAAD V9.15": {
        "color": "#9467BD",
        "band": "#CDB6E0",
    },
}

BUDGET = 1000


RUN_GLOBS = {
    "tsp_construct": {
        "TraceAAD V9.7": ["traceaad_v9_7/v9_7_20260814_150927_*_rep*"],
        "TraceAAD V9.14": ["traceaad_v9_14/v9_14_20260821_001824_*_rep*"],
        "TraceAAD V9.15": ["traceaad_v9_15/v9_15_*_rep*"],
    },
    "cvrp_aco": {
        "TraceAAD V9.7": ["traceaad_v9_7/v9_7_20260814_150927_*_rep*"],
        "TraceAAD V9.14": ["traceaad_v9_14/v9_14_20260821_001824_*_rep*"],
        "TraceAAD V9.15": ["traceaad_v9_15/v9_15_*_rep*"],
    },
    "op_aco": {
        "TraceAAD V9.7": ["traceaad_v9_7/v9_7_20260814_150927_*_rep*"],
        "TraceAAD V9.14": ["traceaad_v9_14/v9_14_20260821_001824_*_rep*"],
        "TraceAAD V9.15": ["traceaad_v9_15/v9_15_*_rep*"],
    },
    "online_bin_packing": {
        "TraceAAD V9.7": ["traceaad_v9_7/v9_7_20260814_150927_*_rep*"],
        "TraceAAD V9.14": ["traceaad_v9_14/v9_14_20260821_001824_*_rep*"],
        "TraceAAD V9.15": ["traceaad_v9_15/v9_15_*_rep*"],
    },
}

# 每个方法的曲线加载器依赖的工件；run 目录必须包含对应文件才计入。
METHOD_ARTIFACTS = {
    "TraceAAD V9.7": "artifacts/candidates.jsonl",
    "TraceAAD V9.14": "evaluations.csv",
    "TraceAAD V9.15": "evaluations.csv",
}


def run_dirs(task: str, method: str) -> list[Path]:
    td = ROOT / "experiments" / task
    dirs: list[Path] = []
    for pattern in RUN_GLOBS[task][method]:
        dirs.extend(td.glob(pattern))
    artifact = METHOD_ARTIFACTS[method]
    return sorted(d for d in dirs if (d / artifact).is_file())


def _v9_points(run: Path) -> list[tuple[int, float]]:
    rows = [json.loads(l) for l in (run / "artifacts" / "candidates.jsonl").read_text().splitlines()]
    pts = []
    for r in rows:
        s = r.get("score", r.get("child_fitness"))
        order = r.get("sample_order", r.get("order"))
        if r.get("status") == "ok" and isinstance(order, int) and isinstance(s, (int, float)):
            pts.append((order, float(s)))
    return pts


def _csv_points(run: Path) -> list[tuple[int, float]]:
    """V9.14+/V9.15 runner: eval_count 为预算轴，fitness 空缺即失败行。"""
    pts: list[tuple[int, float]] = []
    with (run / "evaluations.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            order = row.get("eval_count")
            score = row.get("fitness")
            if order is None or score in (None, ""):
                continue
            try:
                pts.append((int(order), float(score)))
            except ValueError:
                continue
    return pts


def load_curve(run: Path, method: str) -> np.ndarray:
    pts = _csv_points(run) if METHOD_ARTIFACTS[method] == "evaluations.csv" else _v9_points(run)
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
    out = ROOT / "docs" / "experiments" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.75, 3.95))
    handles = []
    all_values = []
    for name, cfg in METHODS.items():
        runs = run_dirs(task, name)
        if not runs:
            raise SystemExit(
                f"{task}/{name}: no runs with {METHOD_ARTIFACTS[name]} under "
                f"{RUN_GLOBS[task][name]}; refusing to write a figure that "
                "silently drops a method"
            )
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
        ncol=3,
        fontsize=8.0,
        handlelength=2.6,
        handletextpad=0.6,
        columnspacing=1.3,
        labelspacing=0.45,
    )
    png = out / f"{task}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return png


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", default=list(TASKS), help="subset of tasks to plot")
    args = ap.parse_args()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    for task in args.tasks:
        if task not in TASKS:
            raise SystemExit(f"unknown task {task}")
        print("wrote", plot_task(task))


if __name__ == "__main__":
    main()

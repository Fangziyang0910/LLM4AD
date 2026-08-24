"""Plot weekly-report search curves: TraceAAD V9.16 vs the five external baselines.

VRPTW baselines kept full per-sample histories (their runs started after the
2026-08-21 artifact cleanup), so all six methods get real best-so-far curves.
On TSP/CVRP/OP/OBP the baselines lost their per-sample histories in that
cleanup; each baseline is drawn as a dashed reference line at its final search
best (mean over 3 runs, read from the same ``results.json`` ``run_records``
train scores as the search-ranking tables). V9.16 reads ``evaluations.csv``
on every task. Internal scores are higher-is-better; figures display the
native task metric (minimize tasks are negated back).

Usage:
    uv run python experiments/plotting/plot_report_curves.py [tasks...]
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

from experiments.analysis.recompute_rankings import (
    ARTIFACTS,
    MCTS_CVRP_BATCH2,
    REPO,
)

BUDGET = 1000

TASKS = [
    "tsp_construct",
    "cvrp_aco",
    "op_aco",
    "online_bin_packing",
    "vrptw_construct",
]

TASK_META = {
    "tsp_construct": ("TSP Construct", False, "Best-so-far tour length (lower is better)"),
    "cvrp_aco": ("CVRP-ACO", False, "Best-so-far route length (lower is better)"),
    "op_aco": ("OP-ACO", True, "Best-so-far collected prize (higher is better)"),
    "online_bin_packing": ("Online Bin Packing", False, "Best-so-far avg. bins (lower is better)"),
    "vrptw_construct": ("VRPTW Construct", False, "Best-so-far total distance (lower is better)"),
}

BASELINES = ["EoH", "ReEvo", "MCTS-AHD", "PathWise", "CALM"]

VRPTW_RUN_GLOBS = {
    "EoH": "eoh/20260822_142500_vrptw_eoh_rep*",
    "ReEvo": "reevo/20260822_142500_vrptw_reevo_rep*",
    "MCTS-AHD": "mcts_ahd/20260822_142500_vrptw_mcts_ahd_rep*",
    "PathWise": "pathwise/20260822_142500_vrptw_pathwise_rep*",
    "CALM": "calm/20260822_142500_vrptw_calm_rep*",
}

COLORS = {
    "TraceAAD V9.16": ("#D62728", "#E8B4B4"),
    "EoH": ("#1F77B4", "#A8CBE3"),
    "ReEvo": ("#8C564B", "#C9A9A3"),
    "MCTS-AHD": ("#FF7F0E", "#FCC07E"),
    "PathWise": ("#9467BD", "#CDB6E0"),
    "CALM": ("#17BECF", "#A7E4EA"),
}


def artifact_rel(task: str, method: str) -> str:
    if task == "cvrp_aco" and method == "MCTS-AHD":
        return MCTS_CVRP_BATCH2
    return ARTIFACTS[task][method][0]


def baseline_final_best(task: str, method: str) -> float:
    path = REPO / "experiments" / task / artifact_rel(task, method) / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = []
    for row in payload.get("run_records", []):
        value = row.get("train_artifact_score")
        if value is None:
            value = row.get("train_best_score")
        if value is not None:
            values.append(float(value))
    if len(values) != 3:
        raise ValueError(f"{task}/{method}: expected 3 train scores in {path}, got {len(values)}")
    return float(np.mean(values))


def curve_from_points(pts: list[tuple[int, float]]) -> np.ndarray:
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


def v916_curve(task: str) -> np.ndarray:
    runs = sorted(
        (REPO / "experiments" / task / "traceaad_v9_16").glob(
            "v9_16_20260822_151700_*_rep*"
        )
    )
    runs = [r for r in runs if (r / "evaluations.csv").is_file()]
    if len(runs) != 3:
        raise SystemExit(f"{task}/V9.16: expected 3 runs with evaluations.csv, got {len(runs)}")
    curves = []
    for run in runs:
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
        curves.append(curve_from_points(pts))
    return np.vstack(curves)


def _sample_points(run: Path) -> list[tuple[int, float]]:
    pts: list[tuple[int, float]] = []
    for f in sorted((run / "logs" / "samples").glob("samples_*.json")):
        for rec in json.loads(f.read_text(encoding="utf-8")):
            order = rec.get("sample_order")
            score = rec.get("score")
            if isinstance(order, int) and isinstance(score, (int, float)):
                pts.append((order, float(score)))
    return pts


def _calm_points(run: Path) -> list[tuple[int, float]]:
    pts: list[tuple[int, float]] = []
    for line in (run / "logs" / "method_events.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        event = json.loads(line)
        if event.get("event") != "epoch":
            continue
        count = event.get("sample_count")
        perf = event.get("best_perf")
        if isinstance(count, int) and isinstance(perf, (int, float)):
            pts.append((count, float(perf)))
    return pts


def vrptw_baseline_curve(method: str) -> np.ndarray:
    runs = sorted((REPO / "experiments" / "vrptw_construct").glob(VRPTW_RUN_GLOBS[method]))
    runs = [r for r in runs if (r / "logs").is_dir()]
    if len(runs) != 3:
        raise SystemExit(f"vrptw_construct/{method}: expected 3 runs, got {len(runs)}")
    loader = _calm_points if method == "CALM" else _sample_points
    curves = [curve_from_points(loader(r)) for r in runs]
    return np.vstack(curves)


def _draw_band_line(ax, name: str, curves: np.ndarray, x: np.ndarray) -> None:
    color, band = COLORS[name]
    ax.fill_between(
        x,
        np.nanmin(curves, axis=0),
        np.nanmax(curves, axis=0),
        step="post",
        color=band,
        alpha=0.3,
        linewidth=0,
    )
    ax.plot(
        x,
        np.nanmean(curves, axis=0),
        drawstyle="steps-post",
        color=color,
        linewidth=2.0,
        label=name,
    )


def _ybounds(values: np.ndarray, refs: list[float], maximize: bool) -> tuple[float, float]:
    if maximize:
        lo = min(float(np.percentile(values, 5)), min(refs, default=np.inf))
        hi = max(float(values.max()), max(refs, default=-np.inf))
    else:
        lo = min(float(values.min()), min(refs, default=np.inf))
        hi = max(float(np.percentile(values, 95)), max(refs, default=-np.inf))
    pad = max((hi - lo) * 0.08, 0.2)
    return (lo - pad * 0.25, hi + pad)


def plot_task(task: str) -> Path:
    _, maximize, ylabel = TASK_META[task]
    out = REPO / "docs" / "reports" / "figures"
    out.mkdir(parents=True, exist_ok=True)

    sign = 1.0 if maximize else -1.0
    x = np.arange(1, BUDGET + 1)
    fig, ax = plt.subplots(figsize=(6.75, 3.95))

    v916 = sign * v916_curve(task)
    _draw_band_line(ax, "TraceAAD V9.16", v916, x)
    refs: list[float] = []

    if task == "vrptw_construct":
        for method in BASELINES:
            curves = sign * vrptw_baseline_curve(method)
            _draw_band_line(ax, method, curves, x)
            refs.append(float(np.nanmean(curves[:, -1])))
        handles, _ = ax.get_legend_handles_labels()
        handles.append(
            Patch(facecolor="#999999", edgecolor="none", alpha=0.3, label="Min–max range across three runs")
        )
    else:
        for method in BASELINES:
            ref = sign * baseline_final_best(task, method)
            refs.append(ref)
            ax.axhline(
                ref,
                color=COLORS[method][0],
                linestyle="--",
                linewidth=1.3,
                alpha=0.9,
                label=f"{method} final best",
            )
        handles, _ = ax.get_legend_handles_labels()
        handles.insert(
            1,
            Patch(facecolor="#999999", edgecolor="none", alpha=0.3, label="Min–max range across three runs"),
        )

    pool = np.concatenate([v916[np.isfinite(v916)], np.array(refs)])
    ax.set(
        xlim=(0, BUDGET),
        ylim=_ybounds(pool, refs, maximize),
        xlabel="Evaluations",
        ylabel=ylabel,
    )
    ax.grid(alpha=0.2)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.97, bottom=0.26)
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(4, len(handles)),
        fontsize=8.0,
        handlelength=2.6,
        handletextpad=0.6,
        columnspacing=1.3,
        labelspacing=0.45,
    )
    png = out / f"{task}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[{task}] V9.16 final best (native): {float(np.nanmean(v916[:, -1])):.6f}")
    for method, ref in zip(BASELINES, refs):
        print(f"[{task}] {method} reference (native): {ref:.6f}")
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

"""Five-task search curves: TraceAAD V9.16 vs five baselines (one figure).

Baselines on TSP/CVRP/OP/OBP read the 20260824_rerun batch; VRPTW reads the
original 20260822 runs. Curves are the mean of three runs, native metric.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BUDGET = 1000
OUT = ROOT / "docs" / "experiments" / "主实验" / "figures"

TASKS = [
    ("tsp_construct", "tsp", "(a) TSP", "Tour length", False),
    ("cvrp_aco", "cvrp", "(b) CVRP", "Route length", False),
    ("op_aco", "op", "(c) OP", "Prize", True),
    ("online_bin_packing", "obp", "(d) OBP", "Bins", False),
    ("vrptw_construct", "vrptw", "(e) VRPTW", "Distance", False),
]

# Okabe–Ito; TraceAAD in vermillion and drawn on top.
METHODS = [
    ("TraceAAD V9.16", "#D55E00", "-"),
    ("EoH", "#0072B2", (0, (4, 1.4))),
    ("ReEvo", "#E69F00", (0, (1, 0.8))),
    ("MCTS-AHD", "#009E73", (0, (2.2, 1.1))),
    ("PathWise", "#CC79A7", (0, (4, 1.2, 1, 1.2))),
    ("CALM", "#56B4E9", (0, (4, 1.2, 1, 1.2, 1, 1.2))),
]
DIRNAME = {
    "EoH": "eoh",
    "ReEvo": "reevo",
    "MCTS-AHD": "mcts_ahd",
    "PathWise": "pathwise",
    "CALM": "calm",
}


def to_curve(pts: list[tuple[int, float]]) -> np.ndarray:
    arr = np.full(BUDGET, -np.inf)
    for n, s in pts:
        if 1 <= n <= BUDGET:
            arr[n - 1] = max(arr[n - 1], s)
    curve = np.maximum.accumulate(arr)
    ok = np.isfinite(curve)
    if ok.any():
        i = np.where(ok)[0]
        curve[i[-1] + 1 :] = curve[i[-1]]
        curve[: i[0]] = np.nan
    return curve


def from_csv(run: Path) -> list[tuple[int, float]]:
    pts = []
    with (run / "evaluations.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            n, s = row.get("eval_count"), row.get("fitness")
            if n and s:
                pts.append((int(n), float(s)))
    return pts


def from_samples(run: Path) -> list[tuple[int, float]]:
    pts = []
    for f in sorted((run / "logs" / "samples").glob("samples_*.json")):
        for rec in json.loads(f.read_text()):
            n, s = rec.get("sample_order"), rec.get("score")
            if isinstance(n, int) and isinstance(s, (int, float)):
                pts.append((n, float(s)))
    return pts


def from_calm(run: Path) -> list[tuple[int, float]]:
    pts = []
    for line in (run / "logs" / "method_events.jsonl").read_text().splitlines():
        ev = json.loads(line)
        if ev.get("event") != "epoch":
            continue
        n, s = ev.get("sample_count"), ev.get("best_perf")
        if isinstance(n, int) and isinstance(s, (int, float)):
            pts.append((n, float(s)))
    return pts


def run_dirs(task: str, short: str, name: str) -> list[Path]:
    if name.startswith("TraceAAD"):
        base = ROOT / "experiments" / task
        pattern = "traceaad_v9_16/v9_16_20260822_151700_*_rep*"
    elif task == "vrptw_construct":
        base = ROOT / "experiments" / task
        m = DIRNAME[name]
        pattern = f"{m}/20260822_142500_vrptw_{m}_rep*"
    else:
        base = ROOT / "experiments" / "其他实验" / "基线重跑-20260824" / task
        m = DIRNAME[name]
        pattern = f"{m}/20260824_rerun_{short}_{m}_rep*"
    dirs = sorted(p for p in base.glob(pattern) if p.is_dir())
    if len(dirs) != 3:
        raise SystemExit(f"{task}/{name}: expected 3 runs, got {len(dirs)} for {pattern}")
    return dirs


def mean_curve(task: str, short: str, name: str) -> np.ndarray:
    curves = []
    for run in run_dirs(task, short, name):
        if name.startswith("TraceAAD"):
            pts = from_csv(run)
        elif name == "CALM":
            pts = from_calm(run)
        else:
            pts = from_samples(run)
        if not pts:
            raise SystemExit(f"no points in {run}")
        curves.append(to_curve(pts))
    stacked = np.vstack(curves)
    count = np.isfinite(stacked).sum(axis=0)
    out = np.full(BUDGET, np.nan)
    ok = count > 0
    out[ok] = np.nansum(stacked[:, ok], axis=0) / count[ok]
    return out


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.35))
    x = np.arange(1, BUDGET + 1)
    for ax, (task, short, title, ylabel, maximize) in zip(axes.ravel(), TASKS):
        sign = 1.0 if maximize else -1.0
        ys = []
        for name, color, ls in METHODS:
            y = sign * mean_curve(task, short, name)
            ys.append(y)
            ax.plot(
                x,
                y,
                color=color,
                linestyle=ls,
                linewidth=2.05 if name.startswith("TraceAAD") else 1.2,
                zorder=5 if name.startswith("TraceAAD") else 2,
                label=name if ax is axes[0, 0] else None,
            )
        window = np.concatenate([y[19:] for y in ys])
        window = window[np.isfinite(window)]
        lo, hi = float(window.min()), float(window.max())
        pad = max((hi - lo) * 0.08, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title, loc="left", fontweight="bold", pad=3)
        ax.set_xlim(0, BUDGET)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Evaluations")
        ax.grid(True, alpha=0.25, linewidth=0.4)
        ax.tick_params(length=2.5, pad=1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1, 2].axis("off")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(
        handles,
        labels,
        loc="center",
        frameon=False,
        handlelength=2.4,
        title="Method",
        title_fontsize=8,
    )
    fig.tight_layout(w_pad=1.1, h_pad=1.0)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "search_curves.png"
    pdf = OUT / "search_curves.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print("wrote", png)
    print("wrote", pdf)


if __name__ == "__main__":
    main()

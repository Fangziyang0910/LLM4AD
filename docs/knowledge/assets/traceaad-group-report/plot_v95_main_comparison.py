#!/usr/bin/env python3
"""Plot TraceAAD V9.5 held-out results against external AAD baselines.

External-baseline values are parsed from the four authoritative task-level result
pages. V9.5 values are read from its formal held-out ``results.json`` files.
Every displayed uncertainty bar is the sample standard deviation across independent
search repeats. TSP/CVRP V9.5 currently have two completed repeats; all other cells
have three.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).resolve().parent

METHODS = ["MCTS-AHD", "PathWise", "EoH", "ReEvo", "CALM", "TraceAAD V9.5"]
BASELINES = METHODS[:-1]
COLORS = {
    "MCTS-AHD": "#243B53",
    "PathWise": "#7B8794",
    "EoH": "#4C72B0",
    "ReEvo": "#9AA5B1",
    "CALM": "#D6A11D",
    "TraceAAD V9.5": "#D95D39",
}
MARKERS = {
    "MCTS-AHD": "o",
    "PathWise": "v",
    "EoH": "^",
    "ReEvo": "D",
    "CALM": "P",
    "TraceAAD V9.5": "s",
}
LINESTYLES = {
    "MCTS-AHD": "-",
    "PathWise": "--",
    "EoH": "-.",
    "ReEvo": ":",
    "CALM": (0, (5, 2)),
    "TraceAAD V9.5": "-",
}

RESULT_PAGES = {
    "TSP": REPO_ROOT / "docs/experiments/tsp_construct/结果汇总.md",
    "CVRP": REPO_ROOT / "docs/experiments/cvrp_aco/结果汇总.md",
    "OP": REPO_ROOT / "docs/experiments/op_aco/结果汇总.md",
    "OBP": REPO_ROOT / "docs/experiments/online_bin_packing/结果汇总.md",
}
V95_RESULTS = {
    "TSP": REPO_ROOT
    / "experiments/tsp_construct/traceaad_v9_5/eval_best_20260812_v95/results.json",
    "CVRP": REPO_ROOT
    / "experiments/cvrp_aco/traceaad_v9_5/eval_best_20260812_v95/results.json",
    "OP": REPO_ROOT
    / "experiments/op_aco/traceaad_v9_5/eval_best_20260812_v95/results.json",
    "OBP": REPO_ROOT
    / "experiments/online_bin_packing/traceaad_v9_5/eval_best_20260812_v95/results.json",
}


def parse_mean_sd(cell: str) -> tuple[float, float]:
    clean = cell.replace("**", "").strip()
    match = re.fullmatch(r"([-+0-9.]+)\s*±\s*([-+0-9.]+)", clean)
    if match is None:
        raise ValueError(f"Cannot parse mean ± SD cell: {cell!r}")
    return float(match.group(1)), float(match.group(2))


def load_baselines(task: str) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    rows: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for line in RESULT_PAGES[task].read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        cells = [part.strip() for part in line.strip().strip("|").split("|")]
        if cells[0] not in BASELINES:
            continue
        parsed = [parse_mean_sd(cell) for cell in cells[1:]]
        rows[cells[0]] = (
            np.array([value[0] for value in parsed]),
            np.array([value[1] for value in parsed]),
            3,
        )
    missing = set(BASELINES) - set(rows)
    if missing:
        raise ValueError(f"Missing baseline rows for {task}: {sorted(missing)}")
    return rows


def load_v95(task: str) -> tuple[np.ndarray, np.ndarray, int]:
    payload = json.loads(V95_RESULTS[task].read_text(encoding="utf-8"))
    if task == "TSP":
        source = payload["eval_results_by_size"]
        keys = ["tsp50", "tsp100", "tsp200"]
    elif task in {"CVRP", "OP"}:
        source = payload["results_by_split"]
        keys = ["test_50", "test_100", "test_200"]
    else:
        source = payload["eval_results_by_scale"]
        keys = ["1k_100", "5k_100", "10k_100", "1k_500", "5k_500", "10k_500"]

    summaries = [source[key]["summary"] for key in keys]
    means = np.array([item["mean_eval_objective"] for item in summaries])
    sds = np.array([item["sample_std_eval_objective"] for item in summaries])
    ns = {int(item["num_runs"]) for item in summaries}
    if len(ns) != 1:
        raise ValueError(f"Inconsistent V9.5 repeat counts for {task}: {sorted(ns)}")
    return means, sds, ns.pop()


def load_all() -> dict[str, dict[str, tuple[np.ndarray, np.ndarray, int]]]:
    data = {task: load_baselines(task) for task in RESULT_PAGES}
    for task in data:
        data[task]["TraceAAD V9.5"] = load_v95(task)
    return data


def normalized_scores(
    task_data: dict[str, tuple[np.ndarray, np.ndarray, int]],
    *,
    higher_is_better: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    means = np.stack([task_data[method][0] for method in METHODS])
    if higher_is_better:
        reference = means.max(axis=0)
        sign = 1.0
    else:
        reference = means.min(axis=0)
        sign = -1.0
    normalized: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for method in METHODS:
        mean, sd, n = task_data[method]
        normalized[method] = (
            sign * (mean - reference) / reference * 100.0,
            sd / reference * 100.0,
            n,
        )
    return normalized


def plot_panel(
    ax: plt.Axes,
    data: dict[str, tuple[np.ndarray, np.ndarray, int]],
    *,
    title: str,
    labels: list[str],
    star_v95: bool = False,
) -> None:
    x = np.arange(len(labels), dtype=float)
    for method in METHODS:
        mean, sd, _n = data[method]
        is_ours = method == "TraceAAD V9.5"
        ax.errorbar(
            x,
            mean,
            yerr=sd,
            label=method,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=2.2 if is_ours else 1.15,
            markersize=5.3 if is_ours else 3.8,
            markeredgecolor="white" if is_ours else COLORS[method],
            markeredgewidth=0.7 if is_ours else 0.2,
            capsize=2.1,
            alpha=1.0 if is_ours else 0.82,
            zorder=6 if is_ours else 3,
        )
    if star_v95:
        ours = data["TraceAAD V9.5"][0]
        for xpos, ypos in zip(x, ours, strict=True):
            ax.annotate(
                "*",
                (xpos, ypos),
                xytext=(5, 2),
                textcoords="offset points",
                color=COLORS["TraceAAD V9.5"],
                fontsize=9,
                fontweight="bold",
            )
    ax.axhline(0, color="#9AA5B1", linewidth=0.75, zorder=1)
    ax.set_title(title, loc="left", pad=7)
    ax.set_xticks(x, labels)
    ax.set_ylabel("距该设置最佳均值的相对差距 (%)\n0 为最佳，越高越好")
    ax.margins(x=0.08)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Noto Serif CJK SC", "Noto Serif CJK JP", "DejaVu Serif"],
            "font.size": 8.5,
            "axes.titlesize": 10.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.4,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 8.0,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.16,
            "grid.linestyle": "-",
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.06,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    raw = load_all()
    normalized = {
        "TSP": normalized_scores(raw["TSP"], higher_is_better=False),
        "CVRP": normalized_scores(raw["CVRP"], higher_is_better=False),
        "OP": normalized_scores(raw["OP"], higher_is_better=True),
        "OBP": normalized_scores(raw["OBP"], higher_is_better=False),
    }

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.8))
    plot_panel(
        axes[0, 0],
        normalized["TSP"],
        title="(a) TSP",
        labels=["50", "100", "200"],
        star_v95=True,
    )
    plot_panel(
        axes[0, 1],
        normalized["CVRP"],
        title="(b) CVRP",
        labels=["50", "100", "200"],
        star_v95=True,
    )
    plot_panel(
        axes[0, 2],
        normalized["OP"],
        title="(c) OP",
        labels=["50", "100", "200"],
    )

    obp_100 = {
        method: (values[0][:3], values[1][:3], values[2])
        for method, values in normalized["OBP"].items()
    }
    obp_500 = {
        method: (values[0][3:], values[1][3:], values[2])
        for method, values in normalized["OBP"].items()
    }
    plot_panel(
        axes[1, 0],
        obp_100,
        title="(d) OBP, capacity=100",
        labels=["1k", "5k", "10k"],
    )
    plot_panel(
        axes[1, 1],
        obp_500,
        title="(e) OBP, capacity=500",
        labels=["1k", "5k", "10k"],
    )

    axes[1, 2].axis("off")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        handlelength=2.8,
        columnspacing=1.2,
        borderaxespad=0,
    )
    axes[1, 2].text(
        0.5,
        0.46,
        "同一任务—尺度内归一化\n误差棒：独立搜索重复间样本标准差",
        ha="center",
        va="center",
        color="#52616B",
        fontsize=8.4,
        linespacing=1.5,
        transform=axes[1, 2].transAxes,
    )
    axes[1, 2].text(
        0.5,
        0.25,
        "*  V9.5 暂为 n=2；其余均为 n=3\n（星号表示重复数，不表示显著性）",
        ha="center",
        va="center",
        color=COLORS["TraceAAD V9.5"],
        fontsize=8.2,
        linespacing=1.45,
        transform=axes[1, 2].transAxes,
    )
    fig.suptitle(
        "TraceAAD V9.5 与代表性 AAD 方法的 held-out 主结果",
        fontsize=13.0,
        fontweight="bold",
        y=0.985,
    )
    fig.supxlabel("测试规模", y=0.015, fontsize=9.0)
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.90,
        bottom=0.10,
        wspace=0.36,
        hspace=0.42,
    )
    fig.savefig(OUTPUT_DIR / "traceaad-v95-main-comparison.pdf")
    fig.savefig(OUTPUT_DIR / "traceaad-v95-main-comparison.png")


if __name__ == "__main__":
    main()

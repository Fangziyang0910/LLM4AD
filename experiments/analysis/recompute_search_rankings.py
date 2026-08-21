"""Generate the formal search-best tables and rankings.

The source of truth is each formal held-out artifact's ``run_records`` block. It
identifies the exact three search runs used by the result pages and records the
best score found on the search evaluator. No held-out objective enters this
report.

Usage:
    uv run python experiments/analysis/recompute_search_rankings.py
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass

from experiments.analysis.recompute_rankings import (
    ARTIFACTS,
    BASELINES,
    MCTS_CVRP_BATCH2,
    REPO,
)

OUTPUT = REPO / "docs" / "experiments" / "搜索结果.md"


@dataclass(frozen=True, slots=True)
class Task:
    key: str
    label: str
    metric: str
    maximize: bool


TASKS = (
    Task("tsp_construct", "TSP Construct", "TSP50 平均 tour length", False),
    Task("cvrp_aco", "CVRP-ACO", "CVRP50 平均最优路径长度", False),
    Task("op_aco", "OP-ACO", "OP50 平均 collected prize", True),
    Task("online_bin_packing", "Online Bin Packing", "训练实例平均箱数", False),
)

EXTRA_ARTIFACTS: dict[str, dict[str, str]] = {
    "V9.1-MCTS": {
        task.key: "traceaad_v9_1/eval_best_20260808_135000" for task in TASKS
    },
    "V9.1-Trajectory": {
        task.key: "traceaad_v9_1/eval_best_20260808_200840" for task in TASKS
    },
    "V9.2": {task.key: "traceaad_v9_2/eval_best_20260809_120403" for task in TASKS},
    "V9.3": {task.key: "traceaad_v9_3/eval_best_20260810_v93" for task in TASKS},
    "V9.4": {task.key: "traceaad_v9_4/eval_best_20260810_v94" for task in TASKS},
    "V9.10": {
        "tsp_construct": "traceaad_v9_10/eval_best_20260818_v910",
        "cvrp_aco": "traceaad_v9_10/eval_best_20260818_v910_cvrp_complete",
        "op_aco": "traceaad_v9_10/eval_best_20260818_v910",
        "online_bin_packing": "traceaad_v9_10/eval_best_20260818_v910",
    },
    "V9.11": {
        "tsp_construct": "traceaad_v9_11/eval_best_20260819_v911_tsp_complete",
        "cvrp_aco": "traceaad_v9_11/eval_best_20260819_v911_cvrp_complete",
        "op_aco": "traceaad_v9_11/eval_best_20260819_v911_op_complete",
        "online_bin_packing": "traceaad_v9_11/eval_best_20260819_1129",
    },
}

MAIN_TRACEAAD = ["V4", "V5", "V8", "V9", "V9.7", "V9.9"]
MAIN_METHODS = BASELINES + MAIN_TRACEAAD
ARCHIVE_TRACEAAD = [
    "V6",
    "V7",
    "V8.2",
    "V8.3",
    "V9.1-MCTS",
    "V9.1-Trajectory",
    "V9.2",
    "V9.3",
    "V9.4",
    "V9.6",
    "V9.8",
    "V9.10",
    "V9.11",
    "V9.12",
    "V9.14",
]
ALL_TRACEAAD = MAIN_TRACEAAD + ARCHIVE_TRACEAAD

DISPLAY = {
    "V4": "TraceAAD V4",
    "V5": "TraceAAD V5",
    "V6": "TraceAAD V6",
    "V7": "TraceAAD V7",
    "V8": "TraceAAD V8",
    "V8.2": "TraceAAD V8.2",
    "V8.3": "TraceAAD V8.3",
    "V9": "TraceAAD V9",
    "V9.1-MCTS": "TraceAAD V9.1 (MCTS-Aligned)",
    "V9.1-Trajectory": "TraceAAD V9.1 (Trajectory)",
    "V9.2": "TraceAAD V9.2",
    "V9.3": "TraceAAD V9.3",
    "V9.4": "TraceAAD V9.4",
    "V9.6": "TraceAAD V9.6",
    "V9.7": "TraceAAD V9.7",
    "V9.8": "TraceAAD V9.8",
    "V9.9": "TraceAAD V9.9",
    "V9.10": "TraceAAD V9.10",
    "V9.11": "TraceAAD V9.11",
    "V9.12": "TraceAAD V9.12",
    "V9.14": "TraceAAD V9.14",
    "CALM": "CALM (w/o GRPO)",
}


def artifact_rel(task: str, method: str) -> str:
    if method in EXTRA_ARTIFACTS:
        return EXTRA_ARTIFACTS[method][task]
    if task == "cvrp_aco" and method == "MCTS-AHD":
        return MCTS_CVRP_BATCH2
    return ARTIFACTS[task][method][0]


def load_scores(task: Task, method: str) -> list[float]:
    rel = artifact_rel(task.key, method)
    path = REPO / "experiments" / task.key / rel / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    values: list[float] = []
    for row in payload.get("run_records", []):
        value = row.get("train_artifact_score")
        if value is None:
            value = row.get("train_best_score")
        if value is not None:
            values.append(float(value))
    if len(values) != 3:
        raise ValueError(f"{task.key}/{method}: expected 3 train scores in {path}, got {len(values)}")
    return values


def objective_values(task: Task, scores: list[float]) -> list[float]:
    return scores if task.maximize else [-score for score in scores]


def mean_objective(task: Task, method: str) -> float:
    return statistics.fmean(objective_values(task, load_scores(task, method)))


def ranks(values: list[float], maximize: bool) -> list[float]:
    order_values = values if maximize else [-value for value in values]
    ordered = sorted(range(len(values)), key=lambda i: -order_values[i])
    result = [0.0] * len(values)
    for position, index in enumerate(ordered, start=1):
        result[index] = float(position)
    for value in set(order_values):
        tied = [i for i, item in enumerate(order_values) if item == value]
        if len(tied) > 1:
            average = statistics.fmean(result[i] for i in tied)
            for i in tied:
                result[i] = average
    return result


def average_ranks(methods: list[str]) -> dict[str, float]:
    totals = {method: 0.0 for method in methods}
    for task in TASKS:
        task_ranks = ranks(
            [mean_objective(task, method) for method in methods],
            task.maximize,
        )
        for method, rank in zip(methods, task_ranks, strict=True):
            totals[method] += rank
    return {method: total / len(TASKS) for method, total in totals.items()}


def format_stat(task: Task, method: str, *, bold: bool = False) -> str:
    values = objective_values(task, load_scores(task, method))
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    text = f"{mean:.6f} ± {std:.6f}"
    return f"**{text}**" if bold else text


def main_table() -> list[str]:
    lines = [
        "## 主表方法",
        "",
        "每个单元格为三次独立搜索的最终 best 均值 +/- 样本标准差。TSP、CVRP、OBP 越低越好，OP 越高越好；加粗为该任务最佳均值。",
        "",
        "| 方法 | TSP Construct | CVRP-ACO | OP-ACO | Online Bin Packing | 平均名次 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    rank = average_ranks(MAIN_METHODS)
    best: dict[str, float] = {}
    for task in TASKS:
        values = [mean_objective(task, method) for method in MAIN_METHODS]
        best[task.key] = max(values) if task.maximize else min(values)
    for method in sorted(MAIN_METHODS, key=lambda item: rank[item]):
        cells = []
        for task in TASKS:
            is_best = mean_objective(task, method) == best[task.key]
            cells.append(format_stat(task, method, bold=is_best))
        lines.append(
            f"| {DISPLAY.get(method, method)} | {' | '.join(cells)} | {rank[method]:.3f} |"
        )
    return lines


def single_version_table() -> list[str]:
    lines = [
        "## TraceAAD 单版本排名",
        "",
        "每次只放入一个 TraceAAD 版本与五个外部对照，在四个搜索任务上排名后取平均名次。该口径不受内部版本数量影响。",
        "",
        "| 版本 | 平均名次 | 六方法中位置 | 相对 MCTS-AHD |",
        "| --- | ---: | ---: | ---: |",
    ]
    rows = []
    for version in ALL_TRACEAAD:
        methods = [version] + BASELINES
        rank = average_ranks(methods)
        position = sorted(methods, key=lambda item: rank[item]).index(version) + 1
        rows.append((version, rank[version], position, rank[version] - rank["MCTS-AHD"]))
    for version, rank, position, delta in sorted(rows, key=lambda row: row[1]):
        lines.append(
            f"| {DISPLAY[version]} | {rank:.3f} | {position}/6 | {delta:+.3f} |"
        )
    return lines


def evidence_table() -> list[str]:
    lines = [
        "## 工件口径",
        "",
        "统计读取正式 held-out `results.json` 的 `run_records`，只使用其中记录的 `train_artifact_score` 或 `train_best_score`。held-out objective 不进入任何数值或排名。",
        "",
        "| 任务 | 搜索指标 | 正式重复 |",
        "| --- | --- | ---: |",
    ]
    for task in TASKS:
        lines.append(f"| {task.label} | {task.metric} | 3 |")
    return lines


def main() -> None:
    sections = [
        "# 搜索结果",
        "",
        "四个任务在统一正式搜索协议下的最终 best 与排名。这里评价方法在给定 evaluator 和 1000 次真实评价预算内找到高质量算法的能力；独立测试与跨规模迁移见[实验结果](实验结果.md)和[归档实验结果](归档实验结果.md)。",
        "",
        *main_table(),
        "",
        *single_version_table(),
        "",
        *evidence_table(),
        "",
    ]
    OUTPUT.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()

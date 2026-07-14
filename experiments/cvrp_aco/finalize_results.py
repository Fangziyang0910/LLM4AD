"""Evaluate all completed CVRP-ACO repeats and write result documents."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = PROJECT_ROOT / "experiments/cvrp_aco/evaluate_best_on_test.py"
PLOTTER = PROJECT_ROOT / "experiments/plotting/plot_cvrp_aco_three_method_search.py"
RESULTS_DIR = PROJECT_ROOT / "docs/results/cvrp_aco"
EVAL_TAG = "eval_20260712_all3"
SPLITS = ("test_50", "test_100")
WORKERS = 16

METHODS: dict[str, dict[str, Any]] = {
    "mcts_ahd": {
        "label": "MCTS-AHD",
        "directory": "mcts_ahd",
        "runs": ("20260711_115024", "20260712_021911", "20260712_021957"),
        "budget": 1000,
    },
    "pathwise": {
        "label": "PathWise",
        "directory": "pathwise",
        "runs": ("20260711_115024", "20260711_192005", "20260711_192010"),
        "budget": 500,
    },
    "traceaad": {
        "label": "TraceAAD",
        "directory": "traceaad",
        "runs": ("20260711_115024", "20260712_041631", "20260712_041658"),
        "budget": 1000,
    },
}


def _run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _run_dir(method: str, run_name: str) -> Path:
    return PROJECT_ROOT / "experiments/cvrp_aco" / METHODS[method]["directory"] / run_name


def _eval_dir(method: str) -> Path:
    return PROJECT_ROOT / "experiments/cvrp_aco" / METHODS[method]["directory"] / EVAL_TAG


def _validate_runs() -> None:
    for method, config in METHODS.items():
        for run_name in config["runs"]:
            run_dir = _run_dir(method, run_name)
            summary_path = run_dir / "logs/run_summary.json"
            if not summary_path.exists():
                raise RuntimeError(f"Missing run summary: {summary_path}")
            summary = json.loads((run_dir / "logs/run_summary.json").read_text(encoding="utf-8"))
            if summary.get("status") != "finished" or summary.get("search_aborted"):
                raise RuntimeError(f"Run is not a completed search: {run_dir} ({summary})")
            if summary.get("num_samples") != config["budget"]:
                raise RuntimeError(
                    f"Unexpected sample count for {run_dir}: "
                    f"{summary.get('num_samples')} != {config['budget']}"
                )


def _run_evaluations() -> None:
    for method, config in METHODS.items():
        output_dir = _eval_dir(method)
        command = [
            sys.executable,
            str(EVALUATOR),
            *(str(_run_dir(method, run_name).relative_to(PROJECT_ROOT)) for run_name in config["runs"]),
            "--output-dir",
            str(output_dir.relative_to(PROJECT_ROOT)),
            "--splits",
            ",".join(SPLITS),
            "--workers",
            str(WORKERS),
        ]
        _run(command)


def _load_payload(method: str) -> dict[str, Any]:
    path = _eval_dir(method) / "results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float | int | None, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _write_result_doc(payloads: dict[str, dict[str, Any]]) -> None:
    first_config = json.loads(
        (_run_dir("mcts_ahd", METHODS["mcts_ahd"]["runs"][0]) / "run_config.json").read_text(
            encoding="utf-8"
        )
    )
    task_eval = first_config.get("task_eval", {})
    lines = [
        "# CVRP-ACO 实验结果",
        "",
        "## 实验参数",
        "",
        "| 项目 | 配置 |",
        "|---|---|",
        f"| 模型 | `{payloads['mcts_ahd']['model']}` |",
        "| 训练集 | `train`：10 个 CVRP50 实例，车辆容量 50 |",
        "| 测试集 | `test_50`、`test_100`：每个 64 个实例 |",
        f"| ACO 参数 | {task_eval.get('n_ants', 30)} ants，{task_eval.get('n_iterations', 100)} iterations，`aco_seed={task_eval.get('aco_seed', 1234)}` |",
        "| LLM 参数 | `temperature=1.0`，`max_tokens=16384` |",
        "| 重复次数 | 每种方法 3 次独立运行 |",
        "| 搜索预算 | MCTS-AHD：1000；PathWise：500；TraceAAD：1000 |",
        "| 方法配置 | MCTS-AHD / PathWise 使用 4 个 evaluators；TraceAAD 使用 `trajectory_ucb`、1 个 evaluator |",
        "| 指标 | `objective` 为平均最优路径长度，越低越好；`score=-objective` |",
        "",
        "## 运行结果",
        "",
        "### 各次运行",
        "",
        "| 方法 | Run | 最优 sample | 训练集 best score | test_50 objective | test_100 objective |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for method, config in METHODS.items():
        for row in payloads[method]["run_records"]:
            test_values = {
                split: next(
                    result["objective"]
                    for result in payloads[method]["results_by_split"][split]["results"]
                    if result["run_name"] == row["run_name"]
                )
                for split in SPLITS
            }
            lines.append(
                f"| {config['label']} | `{row['run_name']}` | {row['best_sample_order']} | "
                f"{_fmt(row['train_best_score'])} | {_fmt(test_values['test_50'])} | {_fmt(test_values['test_100'])} |"
            )

    lines.extend(
        [
            "",
            "### 三次运行平均",
            "",
            "| 方法 | 搜索预算 | test_50 objective | test_100 objective |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, config in METHODS.items():
        summaries = [payloads[method]["results_by_split"][split]["summary"] for split in SPLITS]
        lines.append(
            f"| {config['label']} | {config['budget']} | "
            f"{summaries[0]['mean']:.6f} ± {summaries[0]['sample_std']:.6f} | "
            f"{summaries[1]['mean']:.6f} ± {summaries[1]['sample_std']:.6f} |"
        )

    lines.extend(
        [
            "",
            "![CVRP-ACO 三方法训练曲线](搜索曲线.png)",
            "",
            "## 简单分析",
            "",
            "- MCTS-AHD 在 `test_50` 和 `test_100` 上的平均 objective 都是三种方法中最低的。",
            "- TraceAAD 的结果明显优于 PathWise，但与 MCTS-AHD 相比仍有差距；TraceAAD 在 `test_50` 上的跨 run 波动较小。",
            "- PathWise 只使用 500 次 evaluation，另外两种方法使用 1000 次，因此当前结果不能单独用于判断机制优劣；PathWise 的差距同时受到搜索预算较低的影响。",
            "",
        ]
    )
    path = RESULTS_DIR / "结果汇总.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def main() -> None:
    _validate_runs()
    _run_evaluations()
    _run([sys.executable, str(PLOTTER)])
    payloads = {method: _load_payload(method) for method in METHODS}
    _write_result_doc(payloads)


if __name__ == "__main__":
    main()

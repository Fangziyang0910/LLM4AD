"""Write the MCTS-AHD OP result page from the held-out evaluation artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_PATH = PROJECT_ROOT / "experiments/orienteering_construct/mcts_ahd/eval_best_qwen36_27b_20260714/results.json"
OUTPUT_PATH = PROJECT_ROOT / "docs/results/mcts-ahd-qwen36-27b-orienteering-construct.md"
CURVE_NAME = "mcts-ahd-qwen36-27b-orienteering-construct-search-curve.png"
SIZES = ("op50", "op100", "op200")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.6f}"


def _result_for(payload: dict[str, Any], size: str, run_name: str) -> dict[str, Any]:
    return next(
        row for row in payload["eval_results_by_size"][size]["results"] if row["run_name"] == run_name
    )


def main() -> None:
    payload = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    lines = [
        "# MCTS-AHD Orienteering Construct 实验结果",
        "",
        "## 实验参数",
        "",
        "| 项目 | 配置 |",
        "|---|---|",
        f"| 模型 | `{payload['model']}` |",
        "| 训练集 | OP50，16 个实例，`seed=2024` |",
        "| 测试集 | OP50、OP100、OP200，各 16 个实例，`seed=2025` |",
        "| 重复次数 | 3 次独立运行 |",
        "| 搜索预算 | MCTS-AHD：1000 次评估 |",
        "| 方法配置 | `init_size=4`、`pop_size=10`、`selection_num=2`、4 个 sampler、4 个 evaluator、`alpha=0.5`、`lambda_0=0.1` |",
        "| 指标 | 平均 collected prize，越高越好 |",
            "| 测试方式 | 取每次搜索得到的训练集 best heuristic，在固定 held-out 测试集上完整评估 |",
        "",
        "## 运行结果",
        "",
        "### 各次运行",
        "",
        "| Run | 搜索 artifact | 最优 sample | 操作符 | 训练集 best score | OP50 | OP100 | OP200 |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in payload["run_records"]:
        values = [_result_for(payload, size, row["run_name"])["eval_score"] for size in SIZES]
        lines.append(
            f"| `{row['run_name']}` | `{row['run_dir']}` | {row['best_sample_order']} | "
            f"{row.get('best_operator') or 'n/a'} | {_fmt(row['train_artifact_score'])} | "
            f"{_fmt(values[0])} | {_fmt(values[1])} | {_fmt(values[2])} |"
        )

    lines.extend(
        [
            "",
            "### 三次运行平均",
            "",
            "| 测试规模 | 成功 run 数 | mean ± std |",
            "|---|---:|---:|",
        ]
    )
    for size in SIZES:
        summary = payload["eval_results_by_size"][size]["summary"]
        lines.append(
            f"| `{size.upper()}` | {summary['num_successful_eval_runs']}/{summary['num_runs']} | "
            f"{_fmt(summary['mean_eval_score'])} ± {_fmt(summary['sample_std_eval_score'])} |"
        )

    op50 = payload["eval_results_by_size"]["op50"]["summary"]
    op100 = payload["eval_results_by_size"]["op100"]["summary"]
    all_complete = all(
        payload["eval_results_by_size"][size]["summary"]["num_successful_eval_runs"]
        == payload["eval_results_by_size"][size]["summary"]["num_runs"]
        for size in SIZES
    )
    if not all_complete:
        raise RuntimeError("refusing to write an authoritative result page with incomplete test evaluations")

    lines.extend(
        [
            "",
            f"![MCTS-AHD OP 训练曲线]({CURVE_NAME})",
            "",
            "## Artifact",
            "",
            f"- 测试评估汇总：`{EVAL_PATH.relative_to(PROJECT_ROOT)}`",
            f"- 测试评估使用无 timeout、{payload.get('eval_workers', 'n/a')} 个规模 worker、{payload.get('eval_instance_workers', 'n/a')} 个实例 worker；所有测试规模均须完成后才更新本页。",
            "- 评估命令：`uv run python experiments/orienteering_construct/evaluate_best_on_test.py`",
            "- 绘图命令：`uv run python experiments/plotting/plot_orienteering_construct_mcts_search.py`",
            f"- 训练曲线：`docs/results/{CURVE_NAME}`",
            "- 三个 best heuristic 程序保存在测试评估目录下，与 `results.json` 同目录。",
            "",
            "## 简单分析",
            "",
            f"- OP50 和 OP100 的三个重复均成功完成测试，跨 run 平均 collected prize 分别为 {_fmt(op50['mean_eval_score'])} 和 {_fmt(op100['mean_eval_score'])}。",
            "- 三个测试规模均完成了三个 run 的评估，结果可作为完整三次重复汇总。",
            "- 不同测试规模的 prize 总量不同，分数不应跨 OP50/100/200 直接比较；应在同一规模内比较方法。",
            "",
        ]
    )
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""Generate same-scale independent-test statistics and rankings.

Old TraceAAD versions (V4-V9.12) lost their artifacts in the 2026-08-21
cleanup and are skipped; their rows stay preserved in the committed doc as
the distilled record, so this script prints instead of overwriting the doc.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass

from experiments.analysis.recompute_search_rankings import (
    ALL_TRACEAAD,
    BASELINES,
    DISPLAY,
    MAIN_METHODS,
    REPO,
    artifact_rel,
    has_artifacts,
    ranks,
)

OUTPUT = REPO / "docs" / "experiments" / "同规模测试结果.md"


@dataclass(frozen=True, slots=True)
class Scale:
    task: str
    key: str
    label: str
    maximize: bool


SCALES = (
    Scale("tsp_construct", "tsp50", "TSP50", False),
    Scale("cvrp_aco", "test_50", "CVRP50", False),
    Scale("op_aco", "test_50", "OP50", True),
    Scale("online_bin_packing", "1k_100", "OBP 1k_100", False),
    Scale("online_bin_packing", "5k_100", "OBP 5k_100", False),
    Scale("online_bin_packing", "1k_500", "OBP 1k_500", False),
    Scale("online_bin_packing", "5k_500", "OBP 5k_500", False),
)


def load_values(scale: Scale, method: str) -> list[float]:
    path = REPO / "experiments" / scale.task / artifact_rel(scale.task, method) / "results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if scale.task == "tsp_construct":
        container = payload["eval_results_by_size"]
    elif scale.task == "cvrp_aco" or scale.task == "op_aco":
        container = payload["results_by_split"]
    else:
        container = payload["eval_results_by_scale"]
    rows = container[scale.key]["results"]
    values = []
    for row in rows:
        value = row.get("eval_objective")
        if value is None:
            value = row.get("objective")
        if value is None:
            value = row.get("bins_used_mean")
        if value is not None:
            values.append(float(value))
    if len(values) != 3:
        raise ValueError(f"{scale.task}/{scale.key}/{method}: expected 3 rows, got {len(values)}")
    return values


def format_stat(scale: Scale, method: str, bold: bool = False) -> str:
    values = load_values(scale, method)
    text = f"{statistics.fmean(values):.6f} ± {statistics.stdev(values):.6f}"
    return f"**{text}**" if bold else text


def ranking(methods: list[str]) -> dict[str, float]:
    totals = {method: 0.0 for method in methods}
    for scale in SCALES:
        values = [statistics.fmean(load_values(scale, method)) for method in methods]
        for method, rank in zip(methods, ranks(values, scale.maximize), strict=True):
            totals[method] += rank
    return {method: total / len(SCALES) for method, total in totals.items()}


def build_table(title: str, methods: list[str]) -> list[str]:
    rank = ranking(methods)
    best = {}
    for scale in SCALES:
        values = [statistics.fmean(load_values(scale, method)) for method in methods]
        best[scale] = max(values) if scale.maximize else min(values)
    lines = [
        title,
        "",
        "每个单元格为三次独立同规模测试的均值 ± 样本标准差；加粗为该列最佳均值。平均名次按 7 个同规模设置逐列计算。",
        "",
        "| 方法 | " + " | ".join(scale.label for scale in SCALES) + " | 平均名次 |",
        "| --- | " + " | ".join("---:" for _ in SCALES) + " | ---: |",
    ]
    for method in sorted(methods, key=lambda item: rank[item]):
        cells = []
        for scale in SCALES:
            mean = statistics.fmean(load_values(scale, method))
            cells.append(format_stat(scale, method, mean == best[scale]))
        lines.append(f"| {DISPLAY.get(method, method)} | " + " | ".join(cells) + f" | {rank[method]:.3f} |")
    return lines


def build_single_version_table(versions: list[str]) -> list[str]:
    rows = []
    for version in versions:
        methods = [version] + BASELINES
        rank = ranking(methods)
        position = sorted(methods, key=lambda item: rank[item]).index(version) + 1
        rows.append((version, rank[version], position, rank[version] - rank["MCTS-AHD"]))
    lines = [
        "## TraceAAD 单版本排名",
        "",
        "每次只放入一个 TraceAAD 版本与五个外部基线，避免内部版本数量影响排名。",
        "",
        "| 版本 | 平均名次 | 六方法中位置 | 相对 MCTS-AHD |",
        "| --- | ---: | ---: | ---: |",
    ]
    for version, rank, position, delta in sorted(rows, key=lambda row: row[1]):
        lines.append(f"| {DISPLAY[version]} | {rank:.3f} | {position}/6 | {delta:+.3f} |")
    return lines


def main() -> None:
    available = {m for m in ALL_TRACEAAD + MAIN_METHODS if has_artifacts(m)}
    text = [
        "# 同规模测试结果",
        "",
        "这张表只评价与搜索集规模相同、但实例独立的测试集。它不使用搜索集 best，也不使用跨规模测试：TSP50、CVRP50、OP50，以及 OBP 的 1k/5k × capacity 100/500。OBP 的 10k 设置属于跨规模测试，保留在[实验结果](实验结果.md)。",
        "",
        *build_single_version_table([v for v in ALL_TRACEAAD if v in available]),
        "",
        *build_table("## 主表方法", [m for m in MAIN_METHODS if m in available]),
        "",
        "## 与其他口径的关系",
        "",
        "- 搜索结果：搜索集上 1000 次 evaluator 预算内的最终 best。",
        "- 同规模测试：本页，独立实例上的同规模复核。",
        "- held-out 全表：[实验结果](实验结果.md)，包含同规模与跨规模测试。",
        "",
    ]
    print("\n".join(text))
    print(f"\n(wrote nothing; {OUTPUT.name} keeps pre-cleanup rows — merge manually)")


if __name__ == "__main__":
    main()

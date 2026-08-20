"""从正式测试工件重算跨规模平均名次（代表性同场 / 单版本上场）。

数据源为各任务目录下 `eval_best_*` 的 `results.json`（逐 run 的 eval_objective），
与 `docs/experiments/实验总汇.md` 记录的口径一致。规模共 15 个
（TSP/CVRP/OP 各 3 + OBP 6），每规模在参与方法内排名，并列取平均名次。

整体同场放入四个代表性 TraceAAD（V8、V9、V9.7、V9.8）与 5 个外部对照，共 9 方法。
单版本上场为 1 个 TraceAAD + 5 个外部对照共 6 方法，用于比较全部内部版本。

CVRP 的 MCTS-AHD 官方结果可通过 `--cvrp-mcts batch1|batch2` 切换：
batch1 用 20260711/12 批次（test_50/100 取 eval_20260712_all3，test_200 取
eval_best_20260804_test200）；batch2 用 20260812 批次 eval_best_20260812_cvrp_local。

用法：
    uv run python experiments/analysis/recompute_rankings.py [--cvrp-mcts batch1|batch2]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# 任务 -> (规模键, 方向, 容器键, 数据文件按方法)
TASKS = [
    ("tsp_construct", ["tsp50", "tsp100", "tsp200"], -1, "eval_results_by_size"),
    ("cvrp_aco", ["test_50", "test_100", "test_200"], -1, "results_by_split"),
    ("op_aco", ["test_50", "test_100", "test_200"], +1, "results_by_split"),
    (
        "online_bin_packing",
        ["1k_100", "5k_100", "10k_100", "1k_500", "5k_500", "10k_500"],
        -1,
        "eval_results_by_scale",
    ),
]

# 每任务每方法的工件目录（CVRP 列表按优先级排列，先到先得，全部读取并合并）
ARTIFACTS: dict[str, dict[str, list[str]]] = {
    "tsp_construct": {
        "MCTS-AHD": ["mcts_ahd/eval_best_qwen36_27b_20260710"],
        "PathWise": ["pathwise/eval_best_20260730"],
        "EoH": ["eoh/eval_best_eoh_paper_20260730"],
        "ReEvo": ["reevo/eval_best_20260730"],
        "CALM": ["calm/eval_best_20260807_183000"],
        "V4": ["traceaad_v4/eval_best_20260723_181743"],
        "V5": ["traceaad_v5/eval_best_20260728_151736"],
        "V6": ["traceaad_v6/eval_best_20260802_170400"],
        "V7": ["traceaad_v7/eval_best_20260804_001931"],
        "V8": ["traceaad_v8/eval_best_20260804_173300"],
        "V8.2": ["traceaad_v8/eval_best_20260804_203128"],
        "V8.3": ["traceaad_v8_3/eval_best_20260805_final"],
        "V9": ["traceaad_v9/version9/eval_best_20260807_123753"],
        "V9.6": ["traceaad_v9_6/eval_best_20260812_191011"],
        "V9.7": ["traceaad_v9_7/eval_best_20260815_parentpath"],
        "V9.8": ["traceaad_v9_8/eval_best_20260817_v98_complete"],
        "V9.9": ["traceaad_v9_9/eval_best_20260817_v99_complete"],
        "V9.12": ["traceaad_v9_12/eval_best_20260820_v912_complete"],
    },
    "cvrp_aco": {
        "MCTS-AHD": ["mcts_ahd/eval_20260712_all3", "mcts_ahd/eval_best_20260804_test200"],
        "PathWise": ["pathwise/eval_best_20260730", "pathwise/eval_best_20260804_test200"],
        "EoH": ["eoh/eval_best_eoh_paper_20260730", "eoh/eval_best_20260804_test200"],
        "ReEvo": ["reevo/eval_best_20260730", "reevo/eval_best_20260804_test200"],
        "CALM": ["calm/eval_best_20260807_183000", "calm/eval_best_20260804_test200"],
        "V4": ["traceaad_v4/eval_best_20260723_204526", "traceaad_v4/eval_best_20260804_test200"],
        "V5": ["traceaad_v5/eval_best_20260728_151736", "traceaad_v5/eval_best_20260804_test200"],
        "V6": ["traceaad_v6/eval_best_20260802_170400", "traceaad_v6/eval_best_20260804_test200"],
        "V7": ["traceaad_v7/eval_best_20260804_001931", "traceaad_v7/eval_best_20260804_test200"],
        "V8": ["traceaad_v8/eval_best_20260804_173300"],
        "V8.2": ["traceaad_v8/eval_best_20260804_203128"],
        "V8.3": ["traceaad_v8_3/eval_best_20260805_final", "traceaad_v8_3/eval_best_20260804_test200"],
        "V9": ["traceaad_v9/version9/eval_best_20260807_123753", "traceaad_v9/version9/eval_best_20260804_test200"],
        "V9.6": ["traceaad_v9_6/eval_best_20260812_191011"],
        "V9.7": ["traceaad_v9_7/eval_best_20260815_parentpath"],
        "V9.8": ["traceaad_v9_8/eval_best_20260817_v98_complete"],
        "V9.9": ["traceaad_v9_9/eval_best_20260818_v99_complete"],
        "V9.12": ["traceaad_v9_12/eval_best_20260820_v912_complete"],
    },
    "op_aco": {
        "MCTS-AHD": ["mcts_ahd/eval_best_20260725_104402"],
        "PathWise": ["pathwise/eval_best_20260730"],
        "EoH": ["eoh/eval_best_eoh_paper_20260730"],
        "ReEvo": ["reevo/eval_best_20260730"],
        "CALM": ["calm/eval_best_20260807_183000"],
        "V4": ["traceaad_v4/eval_best_20260723_204526"],
        "V5": ["traceaad_v5/eval_best_20260728_151736"],
        "V6": ["traceaad_v6/eval_best_20260802_170400"],
        "V7": ["traceaad_v7/eval_best_20260804_001931"],
        "V8": ["traceaad_v8/eval_best_20260804_173300"],
        "V8.2": ["traceaad_v8/eval_best_20260804_203128"],
        "V8.3": ["traceaad_v8_3/eval_best_20260805_final"],
        "V9": ["traceaad_v9/version9/eval_best_20260807_123753"],
        "V9.6": ["traceaad_v9_6/eval_best_20260812_191011"],
        "V9.7": ["traceaad_v9_7/eval_best_20260815_parentpath"],
        "V9.8": ["traceaad_v9_8/eval_best_20260817_v98_complete"],
        "V9.9": ["traceaad_v9_9/eval_best_20260817_v99_complete"],
        "V9.12": ["traceaad_v9_12/eval_best_20260820_v912_complete"],
    },
    "online_bin_packing": {
        "MCTS-AHD": ["mcts_ahd/eval_best_20260726_111852"],
        "PathWise": ["pathwise/eval_best_20260730"],
        "EoH": ["eoh/eval_best_eoh_paper_20260730"],
        "ReEvo": ["reevo/eval_best_20260730"],
        "CALM": ["calm/eval_best_20260807_183000"],
        "V4": ["traceaad_v4/eval_best_20260729_230434"],
        "V5": ["traceaad_v5/eval_best_20260728_151736"],
        "V6": ["traceaad_v6/eval_best_20260802_170400"],
        "V7": ["traceaad_v7/eval_best_20260804_001931"],
        "V8": ["traceaad_v8/eval_best_20260804_173300"],
        "V8.2": ["traceaad_v8/eval_best_20260804_203128"],
        "V8.3": ["traceaad_v8_3/eval_best_20260805_final"],
        "V9": ["traceaad_v9/version9/eval_best_20260807_123753"],
        "V9.6": ["traceaad_v9_6/eval_best_20260812_191011"],
        "V9.7": ["traceaad_v9_7/eval_best_20260815_parentpath"],
        "V9.8": ["traceaad_v9_8/eval_best_20260817_v98_complete"],
        "V9.9": ["traceaad_v9_9/eval_best_20260816_v99"],
        "V9.12": ["traceaad_v9_12/eval_best_20260820_v912_complete"],
    },
}

MCTS_CVRP_BATCH2 = "mcts_ahd/eval_best_20260812_cvrp_local"

BASELINES = ["MCTS-AHD", "PathWise", "EoH", "ReEvo", "CALM"]
TRACEAAD_ALL = ["V4", "V5", "V6", "V7", "V8", "V8.2", "V8.3", "V9", "V9.6", "V9.7", "V9.8", "V9.9", "V9.12"]
TRACEAAD_REPRESENTATIVE = ["V8", "V9", "V9.7", "V9.8"]
FIELD = TRACEAAD_REPRESENTATIVE + BASELINES
ALL_METHODS = TRACEAAD_ALL + BASELINES


def load_means(task: str, method: str, cvrp_mcts: str) -> dict[str, float]:
    """返回 {scale_key: 3 次运行 eval_objective 均值}。"""
    if task == "cvrp_aco" and method == "MCTS-AHD":
        artifacts = (
            [MCTS_CVRP_BATCH2]
            if cvrp_mcts == "batch2"
            else ARTIFACTS["cvrp_aco"]["MCTS-AHD"]
        )
    else:
        artifacts = ARTIFACTS[task][method]

    container_key = next(t[3] for t in TASKS if t[0] == task)
    values: dict[str, list[float]] = {}
    for rel in artifacts:
        path = REPO / "experiments" / task / rel / "results.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        container = payload.get(container_key, {})
        for key, block in container.items():
            if key in values:
                continue
            for row in block.get("results", []):
                value = row.get("eval_objective")
                if value is None:
                    value = row.get("objective")
                if value is None:
                    value = row.get("bins_used_mean")
                if value is not None:
                    values.setdefault(key, []).append(float(value))
    if not values:
        raise FileNotFoundError(f"no results for {task}/{method}")
    means = {k: statistics.fmean(v) for k, v in values.items()}
    missing = [k for k in next(t[1] for t in TASKS if t[0] == task) if k not in means]
    if missing:
        raise ValueError(f"{task}/{method} missing scales {missing}: have {sorted(means)}")
    return means


def average_rank(values: list[float], sign: int) -> list[float]:
    score = [v * sign for v in values]
    order = sorted(range(len(score)), key=lambda i: -score[i])
    rank = [0.0] * len(score)
    for i, idx in enumerate(order):
        rank[idx] = i + 1
    for v in set(score):
        idx = [i for i in range(len(score)) if score[i] == v]
        if len(idx) > 1:
            avg = sum(rank[i] for i in idx) / len(idx)
            for i in idx:
                rank[i] = avg
    return rank


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cvrp-mcts", choices=["batch1", "batch2"], default="batch2")
    args = ap.parse_args()

    scales: list[tuple[str, int]] = []
    mu: dict[str, list[float]] = {m: [] for m in ALL_METHODS}
    for task, scale_names, direction, _ in TASKS:
        for m in ALL_METHODS:
            means = load_means(task, m, args.cvrp_mcts)
            mu[m].extend(means[s] for s in scale_names)
        scales.extend((s, direction) for s in scale_names)

    n_scales = len(scales)
    print(f"MCTS-AHD CVRP = {args.cvrp_mcts}；{n_scales} 规模 × {len(FIELD)} 方法（代表性同场）\n")

    rank_sum = {m: 0.0 for m in FIELD}
    for j in range(n_scales):
        r = average_rank([mu[m][j] for m in FIELD], scales[j][1])
        for i, m in enumerate(FIELD):
            rank_sum[m] += r[i]
    field_avg = {m: rank_sum[m] / n_scales for m in FIELD}

    print("=== 1. 代表性方法同场（V8 / V9 / V9.7 / V9.8 + 5 外部对照）===")
    for m in sorted(field_avg, key=lambda m: field_avg[m]):
        print(f"  {m:<12s} {field_avg[m]:.3f}")

    print("\n=== 2. 单版本上场（6 方法同场：1 TraceAAD + 5 外部对照）===")
    print(f"{'版本':<6s} {'平均名次':>8s} {'6方法中':>6s} {'名次差':>7s} {'相对基线优势':>10s}")
    for v in TRACEAAD_ALL:
        field = [v] + BASELINES
        rs = {m: 0.0 for m in field}
        for j in range(n_scales):
            r = average_rank([mu[m][j] for m in field], scales[j][1])
            for i, m in enumerate(field):
                rs[m] += r[i]
        avg = {m: rs[m] / n_scales for m in field}
        sorted_avg = sorted(field, key=lambda m: avg[m])
        position = sorted_avg.index(v) + 1
        mcts = avg["MCTS-AHD"]
        advantage = sum(avg[b] for b in BASELINES) / len(BASELINES) - avg[v]
        print(
            f"  {v:<6s} {avg[v]:8.3f} {position:>4d}   "
            f"{avg[v] - mcts:+7.3f} {advantage:+10.3f}"
        )


if __name__ == "__main__":
    main()

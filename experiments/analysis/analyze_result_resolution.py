"""评估当前 held-out 结果协议的分辨率。

从 `docs/results/实验总汇.md` 解析 11 方法 × 15 held-out 规模的
"均值 ± 样本标准差"（n=3），回答三个问题：

1. 各方法 15 规模平均名次的抽样不确定性有多大？
2. TraceAAD 各版本之间、以及与 MCTS-AHD 之间的差异能否被当前重复数分辨？
3. 方法差异在任务之间怎样分布？

方法：参数化 bootstrap。以报告均值为真值、报告标准差为运行间标准差，
重抽 n=3 个 run 取均值，重算平均名次，得到名次的抽样分布。
该过程只量化"运行间变异经过 3 次重复平均后的残余不确定性"，
不涵盖实例选择、模型端点或实现差异带来的不确定性。

用法：
    uv run python experiments/analysis/analyze_result_resolution.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SUMMARY = REPO / "docs" / "results" / "实验总汇.md"
OUT_DIR = REPO / "docs" / "research" / "version_diagnosis"

N_REP = 3
N_BOOT = 20000
SEED = 20260807

# 结果页中的四张测试表：小节标题 -> (规模列名, 方向, 任务组)
# 方向 +1 表示指标越大越好。
TABLES = [
    ("### 2.1 TSP Construct", ["TSP50", "TSP100", "TSP200"], -1, "TSP"),
    ("### 2.2 CVRP-ACO", ["CVRP50", "CVRP100", "CVRP200"], -1, "CVRP"),
    ("### 2.3 OP-ACO", ["OP50", "OP100", "OP200"], +1, "OP"),
    (
        "### 2.4 Online Bin Packing",
        ["OBP1k/100", "OBP5k/100", "OBP10k/100", "OBP1k/500", "OBP5k/500", "OBP10k/500"],
        -1,
        "OBP",
    ),
]

CELL = re.compile(r"([-\d.]+)\s*±\s*([-\d.]+)")


def parse_summary(text: str) -> tuple[list[str], list[tuple[str, int, str]], np.ndarray, np.ndarray]:
    """解析结果页四张表，返回 (方法列表, 规模元信息, 均值矩阵, 标准差矩阵)。"""
    per_table: list[dict[str, list[tuple[float, float]]]] = []
    scales: list[tuple[str, int, str]] = []

    for heading, scale_names, direction, task in TABLES:
        start = text.index(heading)
        end = text.find("\n### ", start + 1)
        if end == -1:
            end = text.find("\n## ", start + 1)
        block = text[start:end]

        rows: dict[str, list[tuple[float, float]]] = {}
        for line in block.splitlines():
            if not line.startswith("|") or "±" not in line:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            method = cells[0].replace("**", "").replace("TraceAAD ", "").strip()
            values = []
            for c in cells[1:]:
                m = CELL.search(c.replace("**", ""))
                if m is None:
                    raise ValueError(f"无法解析单元格: {c!r}")
                values.append((float(m.group(1)), float(m.group(2))))
            if len(values) != len(scale_names):
                raise ValueError(f"{method} 在 {heading} 的列数不匹配")
            rows[method] = values

        per_table.append(rows)
        scales.extend((n, direction, task) for n in scale_names)

    methods = sorted(
        set.intersection(*(set(r) for r in per_table)),
        key=lambda m: list(per_table[0]).index(m),
    )
    mu = np.array([[v[0] for t in per_table for v in t[m]] for m in methods])
    sd = np.array([[v[1] for t in per_table for v in t[m]] for m in methods])
    return methods, scales, mu, sd


def average_rank(values: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """values: (M, S) -> 每个方法在 S 个规模上的平均名次，并列取平均名次。"""
    score = values * sign
    m, s = score.shape
    ranks = np.empty_like(score)
    for j in range(s):
        col = score[:, j]
        order = (-col).argsort()
        r = np.empty(m)
        r[order] = np.arange(1, m + 1)
        for v in np.unique(col):
            idx = np.flatnonzero(col == v)
            if idx.size > 1:
                r[idx] = r[idx].mean()
        ranks[:, j] = r
    return ranks.mean(axis=1)


def main() -> None:
    methods, scales, mu, sd = parse_summary(SUMMARY.read_text(encoding="utf-8"))
    sign = np.array([d for _, d, _ in scales])
    tasks = [t for _, _, t in scales]
    n_methods, n_scales = mu.shape

    observed = average_rank(mu, sign)

    rng = np.random.default_rng(SEED)
    se = sd / np.sqrt(N_REP)
    boot = np.empty((N_BOOT, n_methods))
    for b in range(N_BOOT):
        boot[b] = average_rank(mu + rng.normal(size=mu.shape) * se, sign)

    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    order = np.argsort(observed)

    print(f"数据源: {SUMMARY.relative_to(REPO)}  ({n_methods} 方法 × {n_scales} 规模, n={N_REP})")
    print(f"参数化 bootstrap: {N_BOOT} 次\n")

    print("=== 1. 15 规模平均名次与抽样不确定性 ===")
    for i in order:
        print(f"  {methods[i]:>10s}  {observed[i]:5.3f}   95% 区间 [{lo[i]:.3f}, {hi[i]:.3f}]")

    ref = methods.index("MCTS-AHD")
    print("\n=== 2. P(该方法平均名次优于 MCTS-AHD) ===")
    p_vs_ref = {}
    for i in order:
        if i == ref:
            continue
        p = float((boot[:, i] < boot[:, ref]).mean())
        p_vs_ref[methods[i]] = p
        print(f"  {methods[i]:>10s}  {p:.3f}")

    versions = [m for m in methods if m.startswith("V")]
    print("\n=== 3. TraceAAD 版本两两 P(行优于列) ===")
    print(" " * 8 + "".join(f"{v:>7s}" for v in versions))
    pairwise = {}
    for a in versions:
        ia = methods.index(a)
        row = f"{a:>7s} "
        for b in versions:
            if a == b:
                row += "      -"
                continue
            p = float((boot[:, ia] < boot[:, methods.index(b)]).mean())
            pairwise[f"{a}>{b}"] = p
            row += f"{p:7.2f}"
        print(row)

    print("\n=== 4. 单版本 vs MCTS-AHD 的逐规模标准化差（正 = TraceAAD 更好）===")
    focus = [v for v in ("V4", "V5", "V6", "V7", "V8", "V8.2", "V8.3") if v in methods]
    print(f"{'scale':>12s}" + "".join(f"{v:>8s}" for v in focus))
    d_table = {v: [] for v in focus}
    for j in range(n_scales):
        row = f"{scales[j][0]:>12s}"
        for v in focus:
            iv = methods.index(v)
            pooled = np.sqrt((sd[iv, j] ** 2 + sd[ref, j] ** 2) / 2)
            d = 0.0 if pooled == 0 else (mu[iv, j] - mu[ref, j]) * sign[j] / pooled
            d_table[v].append(float(d))
            row += f"{d:8.2f}"
        print(row)

    print("\n=== 5. 按任务聚合的平均标准化差 ===")
    task_order = ["TSP", "CVRP", "OP", "OBP"]
    print(f"{'task':>12s}" + "".join(f"{v:>8s}" for v in focus))
    task_means = {}
    for t in task_order:
        idx = [j for j in range(n_scales) if tasks[j] == t]
        row = f"{t:>12s}"
        for v in focus:
            val = float(np.mean([d_table[v][j] for j in idx]))
            task_means.setdefault(v, {})[t] = val
            row += f"{val:8.2f}"
        print(row)
    row = f"{'全部':>12s}"
    overall = {}
    for v in focus:
        overall[v] = float(np.mean(d_table[v]))
        row += f"{overall[v]:8.2f}"
    print(row)

    print("\n=== 6. 运行间变异系数（sd/|mean|，跨 15 规模均值）===")
    cv = {}
    for i in np.argsort([np.mean(sd[i] / np.abs(mu[i])) for i in range(n_methods)]):
        cv[methods[i]] = float(np.mean(sd[i] / np.abs(mu[i])))
        print(f"  {methods[i]:>10s}  {cv[methods[i]] * 100:.3f}%")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": str(SUMMARY.relative_to(REPO)),
        "n_repeats": N_REP,
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "scales": [s for s, _, _ in scales],
        "average_rank": {methods[i]: float(observed[i]) for i in range(n_methods)},
        "average_rank_ci95": {
            methods[i]: [float(lo[i]), float(hi[i])] for i in range(n_methods)
        },
        "p_better_than_mcts_ahd": p_vs_ref,
        "traceaad_pairwise_p": pairwise,
        "cohens_d_vs_mcts_ahd": d_table,
        "cohens_d_by_task": task_means,
        "cohens_d_overall": overall,
        "coefficient_of_variation": cv,
    }
    out = OUT_DIR / "resolution.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()

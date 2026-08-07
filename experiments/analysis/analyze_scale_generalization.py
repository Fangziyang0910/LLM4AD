"""量化各方法产出程序的尺度泛化行为。

问题：一个方法相对 MCTS-AHD 的优势，随 held-out 测试规模变大是增强还是崩塌？

数据：`experiments/<task>/<method_dir>/<eval_dir>/results.json` 中的 held-out 评估。
同一方法的多个 eval 目录按 (task, split, run_name) 合并去重；run_name 不同的
评估目录视为不同方法（例如 V8.3 与 V8.3-credit 是两组独立的 run）。

三个粒度与各自的推断对象（严格区分，避免 instance-level 伪重复）：

1. instance-level（仅描述用）。把某方法 3 个 run 的 instance 向量按元素平均，
   得到"池化程序表现"，再与 MCTS-AHD 的池化表现逐实例配对，报告胜出实例数。
   推断对象是"这几个具体程序在这批具体实例上"的表现。同一程序在 64 个实例上
   的结果不是方法层面的独立重复，因此这里不做显著性检验，也不得用于宣称
   方法之间存在差异。
2. run×run（方法层面）。方法 A 的每个 run 与 MCTS-AHD 的每个 run 两两配对比较
   objective，报告 A 胜出的对数 / 总对数（3×3=9）。推断对象是"随机取一次 A 的
   运行是否优于随机取一次 MCTS-AHD 的运行"，独立单位是 run。
3. run-level（方法层面，主指标）。3 个 run 的 objective 均值差、Welch t 检验
   p 值、Cohen's d，以及各自的 run 间标准差。独立单位是 run，n=3，检验功效很低，
   p 值只用于判断"当前重复数能否分辨"，不用于判断"是否真的没有差异"。

数据可得性：CVRP 与 OP 的评估工件保存了逐实例向量（64 实例），可做 instance-level；
TSP 与 OBP 的工件只保存聚合 objective，instance-level 一栏记为 n/a，
这两个任务的结论只能来自 run×run 与 run-level。

OBP 有 6 个 held-out 配置（物品数 1k/5k/10k × 容量 100/500）。容量不同的目标量纲
不同，因此拆成两条尺度序列分别看趋势。

用法：
    uv run python experiments/analysis/analyze_scale_generalization.py
"""

from __future__ import annotations

import ast
import itertools
import json
import unicodedata
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "docs" / "analysis" / "version_diagnosis"

BASELINE = "MCTS-AHD"

# 尺度序列：视图名 -> (任务目录, 方向, [(split 键, 显示名, 规模数值)])
# 方向 +1 表示 objective 越大越好。
VIEWS: dict[str, tuple[str, int, list[tuple[str, str, int]]]] = {
    "tsp_construct": (
        "tsp_construct",
        -1,
        [("tsp50", "TSP50", 50), ("tsp100", "TSP100", 100), ("tsp200", "TSP200", 200)],
    ),
    "cvrp_aco": (
        "cvrp_aco",
        -1,
        [("test_50", "CVRP50", 50), ("test_100", "CVRP100", 100), ("test_200", "CVRP200", 200)],
    ),
    "op_aco": (
        "op_aco",
        +1,
        [("test_50", "OP50", 50), ("test_100", "OP100", 100), ("test_200", "OP200", 200)],
    ),
    "online_bin_packing@cap100": (
        "online_bin_packing",
        -1,
        [("1k_100", "OBP1k/100", 1000), ("5k_100", "OBP5k/100", 5000), ("10k_100", "OBP10k/100", 10000)],
    ),
    "online_bin_packing@cap500": (
        "online_bin_packing",
        -1,
        [("1k_500", "OBP1k/500", 1000), ("5k_500", "OBP5k/500", 5000), ("10k_500", "OBP10k/500", 10000)],
    ),
}

# 方法显示名 -> {任务目录: [相对 experiments/<task>/ 的 eval 目录]}
# 只登记与结果页一致的权威工件；同一方法目录下 run_name 不同的评估目录另立方法。
METHODS: dict[str, dict[str, list[str]]] = {
    "MCTS-AHD": {
        "tsp_construct": ["mcts_ahd/eval_best_qwen36_27b_20260710"],
        "cvrp_aco": ["mcts_ahd/eval_20260712_all3", "mcts_ahd/eval_best_20260804_test200"],
        "op_aco": ["mcts_ahd/eval_best_20260725_104402"],
        "online_bin_packing": ["mcts_ahd/eval_best_20260726_111852"],
    },
    "EoH": {
        "tsp_construct": ["eoh/eval_best_eoh_paper_20260730"],
        "cvrp_aco": ["eoh/eval_best_eoh_paper_20260730", "eoh/eval_best_20260804_test200"],
        "op_aco": ["eoh/eval_best_eoh_paper_20260730"],
        "online_bin_packing": ["eoh/eval_best_eoh_paper_20260730"],
    },
    "ReEvo": {
        "tsp_construct": ["reevo/eval_best_fair1000_20260730"],
        "cvrp_aco": ["reevo/eval_best_fair1000_20260730", "reevo/eval_best_20260804_test200"],
        "op_aco": ["reevo/eval_best_fair1000_20260730"],
        "online_bin_packing": ["reevo/eval_best_fair1000_20260730"],
    },
    "Pathwise": {
        "tsp_construct": ["pathwise/eval_best_fair1000_20260730"],
        "cvrp_aco": ["pathwise/eval_best_fair1000_20260730", "pathwise/eval_best_20260804_test200"],
        "op_aco": ["pathwise/eval_best_fair1000_20260730"],
        "online_bin_packing": ["pathwise/eval_best_fair1000_20260730"],
    },
    "V4": {
        "tsp_construct": ["traceaad_v4/version4/eval_best_20260723_181743"],
        "cvrp_aco": [
            "traceaad_v4/version4/eval_best_20260723_204526",
            "traceaad_v4/version4/eval_best_20260804_test200",
        ],
        "op_aco": ["traceaad_v4/version4/eval_best_20260723_204526"],
        "online_bin_packing": ["traceaad_v4/version4/eval_best_20260729_230434"],
    },
    "V5": {
        "tsp_construct": ["traceaad_v5/eval_best_20260728_151736"],
        "cvrp_aco": ["traceaad_v5/eval_best_20260728_151736", "traceaad_v5/eval_best_20260804_test200"],
        "op_aco": ["traceaad_v5/eval_best_20260728_151736"],
        "online_bin_packing": ["traceaad_v5/eval_best_20260728_151736"],
    },
    "V6": {
        "tsp_construct": ["traceaad_v6/eval_best_20260803"],
        "cvrp_aco": ["traceaad_v6/eval_best_20260803", "traceaad_v6/eval_best_20260804_test200"],
        "op_aco": ["traceaad_v6/eval_best_20260803"],
        "online_bin_packing": ["traceaad_v6/eval_best_20260803"],
    },
    "V7": {
        "tsp_construct": ["traceaad_v7/eval_best_20260804"],
        "cvrp_aco": ["traceaad_v7/eval_best_20260804", "traceaad_v7/eval_best_20260804_test200"],
        "op_aco": ["traceaad_v7/eval_best_20260804"],
        "online_bin_packing": ["traceaad_v7/eval_best_20260804"],
    },
    "V8": {
        "tsp_construct": ["traceaad_v8/eval_best_20260805"],
        "cvrp_aco": ["traceaad_v8/eval_best_20260805"],
        "op_aco": ["traceaad_v8/eval_best_20260805"],
        "online_bin_packing": ["traceaad_v8/eval_best_20260805"],
    },
    "V8.2": {
        "tsp_construct": ["traceaad_v8/eval_best_v82_20260805"],
        "cvrp_aco": ["traceaad_v8/eval_best_v82_20260805"],
        "op_aco": ["traceaad_v8/eval_best_v82_20260805"],
        "online_bin_packing": ["traceaad_v8/eval_best_v82_20260805"],
    },
    "V8.3": {
        "tsp_construct": ["traceaad_v8_3/eval_best_v83_20260806"],
        "cvrp_aco": ["traceaad_v8_3/eval_best_v83_20260806"],
        "op_aco": ["traceaad_v8_3/eval_best_v83_20260806"],
        "online_bin_packing": ["traceaad_v8_3/eval_best_v83_20260806"],
    },
    "V8.3-credit": {
        "tsp_construct": ["traceaad_v8_3/eval_best_v83_credit_20260807"],
        "op_aco": ["traceaad_v8_3/eval_best_v83_credit_20260807"],
        "online_bin_packing": ["traceaad_v8_3/eval_best_v83_credit_20260807"],
    },
}

# 三种工件布局的 split 容器键
SPLIT_CONTAINERS = ("results_by_split", "eval_results_by_size", "eval_results_by_scale")
# 三种工件布局的 objective 字段（越大越好 / 越小越好由 VIEWS 的方向决定）
OBJ_FIELDS = ("objective", "eval_objective", "bins_used_mean")
INSTANCE_FIELDS = ("instance_costs", "instance_prizes")

# 判定"崩塌 / 增强"的幅度门槛
WIN_DELTA_THRESHOLD = 0.15  # instance 胜率绝对变化（百分点）
WIN_RELATIVE_THRESHOLD = 0.5  # 单调时的相对变化，用于胜率贴近 0 的情形
WIN_FLOOR = 0.05  # 起点胜率低于此值时相对变化没有意义，退回用 Cohen's d 判定
D_DELTA_THRESHOLD = 0.5  # Cohen's d 绝对变化

# 已知参考值，用于校验数据加载正确：
# (视图, 方法, split) -> (instance 胜出数, run×run 胜出数 或 None, Welch p 或 None,
#                          逐 run instance 胜率百分数 或 None)
REFERENCE_CHECKS = [
    ("cvrp_aco", "V5", "test_50", 22, None, None, None),
    ("cvrp_aco", "V5", "test_100", 4, None, None, None),
    ("cvrp_aco", "V5", "test_200", 0, 0, 0.076, None),
    ("cvrp_aco", "V8", "test_50", 34, None, None, None),
    ("cvrp_aco", "V8", "test_100", 11, None, None, None),
    ("cvrp_aco", "V8", "test_200", 0, 2, 0.266, [17, 0, 38]),
    ("op_aco", "V5", "test_50", 34, None, None, None),
    ("op_aco", "V5", "test_100", 39, None, None, None),
    ("op_aco", "V5", "test_200", 55, 5, 0.444, None),
    ("op_aco", "V8", "test_50", 17, None, None, None),
    ("op_aco", "V8", "test_100", 19, None, None, None),
    ("op_aco", "V8", "test_200", 12, None, None, [67, 12, 14]),
]


def read_eval(path: Path) -> dict:
    """读取一个 results.json，返回 {split: [记录]}。

    早期工件把嵌套结构存成 Python repr 字符串，这里统一还原。
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in SPLIT_CONTAINERS:
        if key in raw:
            value = raw[key]
            return ast.literal_eval(value) if isinstance(value, str) else value
    raise ValueError(f"{path} 中找不到 split 容器")


def collect(task: str, eval_dirs: list[str]) -> dict[str, dict[str, dict]]:
    """合并一个方法在一个任务下的多个 eval 目录，返回 {split: {run_name: 记录}}。"""
    merged: dict[str, dict[str, dict]] = {}
    for rel in eval_dirs:
        path = REPO / "experiments" / task / rel / "results.json"
        if not path.exists():
            continue
        for split, block in read_eval(path).items():
            bucket = merged.setdefault(split, {})
            for record in block["results"]:
                name = record["run_name"]
                if name in bucket:  # 按 (task, split, run_name) 去重
                    continue
                obj = next((record[f] for f in OBJ_FIELDS if f in record), None)
                vec = next((record[f] for f in INSTANCE_FIELDS if f in record), None)
                bucket[name] = {
                    "objective": float(obj),
                    "instances": None if vec is None else np.asarray(vec, dtype=float),
                }
    return merged


def load_all() -> dict[str, dict[str, dict[str, dict[str, dict]]]]:
    """返回 {方法: {任务: {split: {run_name: 记录}}}}。"""
    return {
        method: {task: collect(task, dirs) for task, dirs in per_task.items()}
        for method, per_task in METHODS.items()
    }


def cohens_d(a: np.ndarray, b: np.ndarray, sign: int) -> float:
    """正值表示 a 比 b 好。pooled 标准差为 0 时返回 0。"""
    sa, sb = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt((sa**2 + sb**2) / 2)
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) * sign / pooled)


def compare(cand: dict[str, dict], base: dict[str, dict], sign: int) -> dict | None:
    """在一个 split 上比较候选方法与基线，返回三个粒度的统计量。"""
    if not cand or not base:
        return None

    cand_obj = np.array([r["objective"] for r in cand.values()])
    base_obj = np.array([r["objective"] for r in base.values()])

    # 粒度 2：run×run 两两配对
    pairs = list(itertools.product(cand_obj, base_obj))
    rr_win = sum(1 for x, y in pairs if (x - y) * sign > 0)

    # 粒度 3：run-level
    if len(cand_obj) > 1 and len(base_obj) > 1:
        with warnings.catch_warnings():  # 两组几乎完全相同时 scipy 会报精度损失
            warnings.simplefilter("ignore", RuntimeWarning)
            p_value = float(stats.ttest_ind(cand_obj, base_obj, equal_var=False).pvalue)
    else:
        p_value = float("nan")

    out = {
        "n_runs_cand": len(cand_obj),
        "n_runs_base": len(base_obj),
        "mean_cand": float(cand_obj.mean()),
        "mean_base": float(base_obj.mean()),
        "signed_gap": float((cand_obj.mean() - base_obj.mean()) * sign),
        "sd_cand": float(cand_obj.std(ddof=1)),
        "sd_base": float(base_obj.std(ddof=1)),
        "welch_p": p_value,
        "cohens_d": cohens_d(cand_obj, base_obj, sign),
        "runxrun_win": rr_win,
        "runxrun_total": len(pairs),
        "instance_available": False,
        "instance_win": None,
        "instance_total": None,
        "instance_win_rate": None,
        "per_run_instance_win_rate": None,
    }

    cand_vecs = [r["instances"] for r in cand.values()]
    base_vecs = [r["instances"] for r in base.values()]
    if any(v is None for v in cand_vecs + base_vecs):
        return out
    lengths = {len(v) for v in cand_vecs + base_vecs}
    if len(lengths) != 1:
        return out

    # 粒度 1：池化程序表现逐实例配对（仅描述，不做检验）
    pooled_cand = np.mean(cand_vecs, axis=0)
    pooled_base = np.mean(base_vecs, axis=0)
    win = int((((pooled_cand - pooled_base) * sign) > 0).sum())
    out.update(
        instance_available=True,
        instance_win=win,
        instance_total=len(pooled_base),
        instance_win_rate=win / len(pooled_base),
        per_run_instance_win_rate=[
            float(((v - pooled_base) * sign > 0).mean()) for v in cand_vecs
        ],
    )
    return out


def clean_seq(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]


def trend(values: list[float | None]) -> str:
    """序列的单调性。并列不打断单调性，但全部并列记为持平。"""
    clean = clean_seq(values)
    if len(clean) < 2:
        return "数据不足"
    diffs = np.diff(clean)
    if np.all(diffs == 0):
        return "持平"
    if np.all(diffs >= 0):
        return "单调上升"
    if np.all(diffs <= 0):
        return "单调下降"
    return "无单调趋势"


def label(win_seq: list[float | None], d_seq: list[float | None]) -> tuple[str, str]:
    """给出尺度行为标签与判定依据。

    只在有逐实例向量时给出尺度行为标签。instance 胜率由 64 个实例决定，
    跨尺度的单调性可以与逐 run 的同向性互相印证，方向判断是可靠的描述。

    缺少逐实例向量时（TSP 与 OBP 的评估工件只保存聚合 objective），
    唯一可用的序列是 3 次重复算出的 Cohen's d。该 d 在 n=3 下的标准误
    与常见的跨尺度差同量级，其单调性无法与噪声区分，因此不给标签。
    d 序列仍然输出，供人工判断。
    """
    wins = clean_seq(win_seq)
    if len(wins) < 2:
        return "证据不足", "无"
    if wins[0] < WIN_FLOOR:
        return "地板效应", "instance胜率"

    delta = wins[-1] - wins[0]
    direction = trend(win_seq)
    strong = abs(delta) >= WIN_DELTA_THRESHOLD or (
        direction in ("单调上升", "单调下降")
        and abs(delta) / wins[0] >= WIN_RELATIVE_THRESHOLD
    )
    if strong and delta < 0:
        return ("尺度崩塌" if direction == "单调下降" else "整体下滑(非单调)"), "instance胜率"
    if strong and delta > 0:
        return ("尺度增强" if direction == "单调上升" else "整体上升(非单调)"), "instance胜率"
    return "无明显趋势", "instance胜率"


def width(text: str) -> int:
    """终端显示宽度，中文字符按 2 列计。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def pad(text: str, w: int, align: str = "<") -> str:
    fill = max(0, w - width(text))
    if align == "<":
        return text + " " * fill
    return " " * fill + text


def fmt(value: float | None, spec: str, na: str = "n/a") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return na
    return format(value, spec)


def row(cells: list[tuple[str, int, str]]) -> str:
    return "  " + "".join(pad(t, w, a) for t, w, a in cells)


def analyze() -> dict:
    """计算全部视图 × 方法 × 尺度的三粒度统计量与尺度趋势。"""
    data = load_all()
    payload: dict = {
        "baseline": BASELINE,
        "granularity_notes": {
            "instance_level": "3 个 run 的 instance 向量按元素平均后逐实例配对；"
            "推断对象是这几个具体程序在这批实例上的表现；伪重复，仅描述，不做显著性检验",
            "run_by_run": "候选 3 run × 基线 3 run 的 objective 两两配对；"
            "推断对象是方法层面随机一次运行的相对优劣",
            "run_level": "n=3 的 objective 均值差 / Welch t 检验 / Cohen's d；"
            "推断对象是方法层面的期望表现，功效很低",
        },
        "thresholds": {
            "instance_win_rate_delta": WIN_DELTA_THRESHOLD,
            "instance_win_rate_relative": WIN_RELATIVE_THRESHOLD,
            "instance_win_rate_floor": WIN_FLOOR,
            "cohens_d_delta": D_DELTA_THRESHOLD,
        },
        "views": {},
    }

    for view, (task, sign, scales) in VIEWS.items():
        base_by_split = data[BASELINE].get(task, {})
        has_instance = any(
            r["instances"] is not None
            for split, _, _ in scales
            for r in base_by_split.get(split, {}).values()
        )
        view_payload: dict = {
            "task": task,
            "direction": sign,
            "scales": [{"split": s, "label": d, "size": n} for s, d, n in scales],
            "instance_level_available": has_instance,
            "methods": {},
        }

        for method in METHODS:
            if method == BASELINE:
                continue
            cand_by_split = data[method].get(task, {})
            if not cand_by_split:
                continue
            per_scale = {}
            for split, disp, size in scales:
                res = compare(cand_by_split.get(split, {}), base_by_split.get(split, {}), sign)
                if res is not None:
                    per_scale[split] = {"label": disp, "size": size, **res}
            if not per_scale:
                continue

            ordered = [per_scale[s] for s, _, _ in scales if s in per_scale]
            win_seq = [r["instance_win_rate"] for r in ordered]
            rr_seq = [r["runxrun_win"] / r["runxrun_total"] for r in ordered]
            d_seq = [r["cohens_d"] for r in ordered]
            tag, basis = label(win_seq, d_seq)
            wins_clean = clean_seq(win_seq)

            view_payload["methods"][method] = {
                "by_scale": per_scale,
                "trend": {
                    "instance_win_rate": trend(win_seq),
                    "runxrun_win_rate": trend(rr_seq),
                    "cohens_d": trend(d_seq),
                    "instance_win_rate_delta": (
                        wins_clean[-1] - wins_clean[0] if len(wins_clean) >= 2 else None
                    ),
                    "cohens_d_delta": d_seq[-1] - d_seq[0],
                    "label": tag,
                    "label_basis": basis,
                },
            }
        payload["views"][view] = view_payload
    return payload


def verify(payload: dict) -> None:
    """用已知参考值校验数据加载与统计口径。"""
    problems = []
    for view, method, split, inst, rr, p, per_run in REFERENCE_CHECKS:
        got = payload["views"][view]["methods"][method]["by_scale"][split]
        if got["instance_win"] != inst:
            problems.append(f"{view}/{method}/{split} instance {got['instance_win']} != {inst}")
        if rr is not None and got["runxrun_win"] != rr:
            problems.append(f"{view}/{method}/{split} run×run {got['runxrun_win']} != {rr}")
        if p is not None and abs(got["welch_p"] - p) > 5e-4:
            problems.append(f"{view}/{method}/{split} p {got['welch_p']:.3f} != {p}")
        if per_run is not None:
            got_per = [round(v * 100) for v in got["per_run_instance_win_rate"]]
            if got_per != per_run:
                problems.append(f"{view}/{method}/{split} 逐 run {got_per} != {per_run}")
    if problems:
        raise AssertionError("参考值校验失败:\n  " + "\n  ".join(problems))
    print(f"参考值校验通过（{len(REFERENCE_CHECKS)} 项）\n")


def report(payload: dict) -> None:
    for view, (task, sign, scales) in VIEWS.items():
        view_payload = payload["views"][view]
        methods = view_payload["methods"]
        print("=" * 108)
        arrow = "越大越好" if sign > 0 else "越小越好"
        print(f"任务视图 {view}（{arrow}）：{' -> '.join(d for _, d, _ in scales)}")
        if not view_payload["instance_level_available"]:
            print("  注：该任务的评估工件只保存聚合 objective，没有逐实例向量，instance-level 记为 n/a；")
            print("      尺度行为只能由 Cohen's d 判定，而 d 在 n=3 下本身噪声很大，结论强度弱于 CVRP/OP")

        # ---- 表 1：逐尺度三粒度 ----
        cols = [("方法", 14, "<"), ("尺度", 12, ">"), ("inst胜", 10, ">"), ("inst%", 8, ">"),
                ("run×run", 10, ">"), ("均值差", 11, ">"), ("d", 8, ">"), ("p", 8, ">"),
                ("sd_A", 9, ">"), ("sd_M", 9, ">")]
        print(row(cols))
        print("  " + "-" * sum(c[1] for c in cols))
        for method, blob in methods.items():
            for split, _, _ in scales:
                res = blob["by_scale"].get(split)
                if res is None:
                    continue
                inst = f"{res['instance_win']}/{res['instance_total']}" if res["instance_available"] else "n/a"
                pct = f"{res['instance_win_rate'] * 100:.0f}%" if res["instance_available"] else "n/a"
                print(row([
                    (method, 14, "<"), (res["label"], 12, ">"), (inst, 10, ">"), (pct, 8, ">"),
                    (f"{res['runxrun_win']}/{res['runxrun_total']}", 10, ">"),
                    (f"{res['signed_gap']:+.3f}", 11, ">"), (f"{res['cohens_d']:+.2f}", 8, ">"),
                    (fmt(res["welch_p"], ".3f"), 8, ">"),
                    (f"{res['sd_cand']:.3f}", 9, ">"), (f"{res['sd_base']:.3f}", 9, ">"),
                ]))

        # ---- 表 2：尺度趋势 ----
        print()
        print(row([("方法", 14, "<"), ("inst胜率序列", 22, ">"), ("run×run序列", 20, ">"),
                   ("d序列", 22, ">"), ("  尺度行为(判据)", 0, "<")]))
        for method, blob in methods.items():
            by_scale = blob["by_scale"]
            present = [by_scale[s] for s, _, _ in scales if s in by_scale]
            wins = "  ".join(fmt(r["instance_win_rate"], ".0%") for r in present)
            rrs = "  ".join(f"{r['runxrun_win']}/{r['runxrun_total']}" for r in present)
            ds = "  ".join(f"{r['cohens_d']:+.2f}" for r in present)
            tr = blob["trend"]
            if tr["label_basis"] == "instance胜率":
                note = (f"{tr['label']}（instance 胜率 {tr['instance_win_rate']}, "
                        f"Δ={tr['instance_win_rate_delta']:+.0%}）")
            else:
                note = (f"{tr['label']}；d 序列 {tr['cohens_d']}, Δd={tr['cohens_d_delta']:+.2f}"
                        f"（n=3，仅供参考，不作判定）")
            print(row([(method, 14, "<"), (wins, 22, ">"), (rrs, 20, ">"), (ds, 22, ">"),
                       ("  " + note, 0, "<")]))

        # ---- 表 3：逐 run instance 胜率 ----
        if view_payload["instance_level_available"]:
            print()
            print("  逐 run 的 instance 胜率（每个 run 对 MCTS-AHD 池化表现；伪重复, 仅描述）")
            print(row([("方法", 14, "<")] + [(d, 24, ">") for _, d, _ in scales]))
            for method, blob in methods.items():
                cells = [(method, 14, "<")]
                for split, _, _ in scales:
                    res = blob["by_scale"].get(split)
                    rates = res["per_run_instance_win_rate"] if res else None
                    text = "n/a" if not rates else " / ".join(f"{v * 100:.0f}%" for v in rates)
                    cells.append((text, 24, ">"))
                print(row(cells))
        print()

    # ---- 汇总 ----
    view_names = list(VIEWS)
    short = [v.replace("online_bin_packing@", "OBP-").replace("_construct", "").replace("_aco", "")
             for v in view_names]
    print("=" * 108)
    print("尺度行为汇总（相对 MCTS-AHD）")
    print("  标签只在有逐实例向量时给出。证据不足 = 评估工件只有聚合 objective；"
          "地板效应 = 各尺度胜率都近 0，方向无从谈起")
    print(row([("方法", 14, "<")] + [(s, 20, ">") for s in short]))
    for method in METHODS:
        if method == BASELINE:
            continue
        cells = [(method, 14, "<")]
        for view in view_names:
            blob = payload["views"][view]["methods"].get(method)
            cells.append(((blob["trend"]["label"] if blob else "缺数据"), 20, ">"))
        print(row(cells))

    print()
    print("run-level 可分辨性：Welch p < 0.05 的 (视图×尺度) 计数，以及 15 个尺度的平均 Cohen's d")
    print(row([("方法", 14, "<"), ("p<0.05", 12, ">"), ("其中优于基线", 16, ">"), ("平均 d", 12, ">")]))
    for method in METHODS:
        if method == BASELINE:
            continue
        sig = fav = total = 0
        ds = []
        for view in view_names:
            blob = payload["views"][view]["methods"].get(method)
            if not blob:
                continue
            for res in blob["by_scale"].values():
                total += 1
                ds.append(res["cohens_d"])
                if not np.isnan(res["welch_p"]) and res["welch_p"] < 0.05:
                    sig += 1
                    fav += res["signed_gap"] > 0
        print(row([(method, 14, "<"), (f"{sig}/{total}", 12, ">"), (str(fav), 16, ">"),
                   (f"{np.mean(ds):+.2f}", 12, ">")]))


def main() -> None:
    payload = analyze()
    verify(payload)
    report(payload)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "scale_generalization.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出 {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()


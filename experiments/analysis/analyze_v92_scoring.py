"""用既有 V9 / V9.1 工件识别 V9.2 轨迹评分信号。

本脚本只使用一次预算分配发生前已经存在的节点、边和历史选择结果，避免把
当前批次或未来后代泄漏进特征。它回答三个探索性问题：

1. 锚点 fitness、形成窗口形态、已测试直接分支和既往预算结果，哪些能预测
   下一次生成的绝对质量、全局推进与路线推进；
2. 在按 run 留一的测试中，轨迹特征是否比只看锚点 fitness 提供稳定增量；
3. 一次直接退步之后，经后续真实扩展产生短程突破的现象有多常见。

它不能回答“如果当时选择另一个锚点或另一条窗口会怎样”，因为历史策略只
观察了实际被选项的结果。输出因此是评分候选的探索性证据，不是离线策略收益。

用法：

    uv run python experiments/analysis/analyze_v92_scoring.py
    uv run python experiments/analysis/analyze_v92_scoring.py --json-out /tmp/v92.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
EPS = 1e-6
WINDOW_NODES = 8
DELAYED_DEPTH = 3

PROTOCOLS = {
    "traceaad-v9-core": "V9",
    "traceaad-v9.1-mcts-aligned": "V9.1-old",
    "traceaad-v9.1-trajectory-centered": "V9.1-trajectory",
}


@dataclass(frozen=True)
class Node:
    node_id: int
    parent_id: int
    q: float
    sample_order: int
    creation_batch: int


@dataclass(frozen=True)
class Edge:
    parent_id: int
    child_id: int
    batch_id: int
    q: float
    delta_parent: float


def load_jsonl(path: Path) -> tuple[list[dict], int]:
    """读取可追加 JSONL；只允许忽略正在写入的最后一条不完整记录。"""
    rows: list[dict] = []
    skipped = 0
    if not path.is_file():
        return rows, skipped
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
            skipped += 1
    return rows, skipped


def discover_runs() -> list[Path]:
    paths = list(REPO.glob("experiments/*/traceaad_v9/version9/*/run_config.json"))
    paths += list(REPO.glob("experiments/*/traceaad_v9_1/*/run_config.json"))
    runs = []
    for config_path in sorted(paths):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        protocol = config.get("method_params", {}).get("protocol_id")
        if protocol in PROTOCOLS:
            runs.append(config_path.parent)
    return runs


def finite_number(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, np.asarray(values, dtype=float), 1)[0])


def sign_change_rate(deltas: list[float]) -> float:
    signs = [1 if x > EPS else -1 if x < -EPS else 0 for x in deltas]
    signs = [x for x in signs if x]
    if len(signs) < 2:
        return 0.0
    return sum(a != b for a, b in zip(signs, signs[1:])) / (len(signs) - 1)


def path_to(node_id: int, nodes: dict[int, Node]) -> list[int]:
    path: list[int] = []
    current = node_id
    seen: set[int] = set()
    while current != -1:
        if current in seen or current not in nodes:
            raise ValueError(f"invalid parent chain at node {current}")
        seen.add(current)
        path.append(current)
        current = nodes[current].parent_id
    return list(reversed(path))


def run_status(run_dir: Path) -> str:
    summary = run_dir / "logs" / "summary.json"
    if not summary.is_file():
        return "running_or_incomplete"
    try:
        return str(json.loads(summary.read_text(encoding="utf-8")).get("status"))
    except (json.JSONDecodeError, OSError):
        return "unreadable"


def parse_run(run_dir: Path) -> tuple[list[dict], dict]:
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    params = config.get("method_params", {})
    protocol_id = params["protocol_id"]
    protocol = PROTOCOLS[protocol_id]
    maximize = bool(params.get("maximize", True))
    budget = int(params.get("max_sample_nums", 1000))
    task = str(config.get("task", run_dir.parts[-4]))

    candidates, skipped_candidates = load_jsonl(run_dir / "artifacts/candidates.jsonl")
    decisions, skipped_decisions = load_jsonl(run_dir / "artifacts/decisions.jsonl")
    edge_rows, skipped_edges = load_jsonl(run_dir / "artifacts/edges.jsonl")

    sample_q: dict[int, float] = {}
    candidates_by_batch: dict[int, list[dict]] = defaultdict(list)
    for row in candidates:
        batch = row.get("batch_id")
        if batch is not None:
            candidates_by_batch[int(batch)].append(row)
        if row.get("sample_order") is None or not finite_number(row.get("score")):
            continue
        score = float(row["score"])
        sample_q[int(row["sample_order"])] = score if maximize else -score

    nodes: dict[int, Node] = {}
    for row in decisions:
        if row.get("event") != "initial_node_created":
            continue
        order = int(row["sample_order"])
        if order not in sample_q:
            continue
        node_id = int(row["node_id"])
        nodes[node_id] = Node(node_id, -1, sample_q[order], order, 0)

    edges: list[Edge] = []
    edge_by_child: dict[int, Edge] = {}
    for row in edge_rows:
        order = int(row["sample_order"])
        if order not in sample_q:
            continue
        edge = Edge(
            parent_id=int(row["parent_id"]),
            child_id=int(row["child_id"]),
            batch_id=int(row["batch_id"]),
            q=sample_q[order],
            delta_parent=float(row["delta_parent"]),
        )
        edges.append(edge)
        edge_by_child[edge.child_id] = edge
        nodes[edge.child_id] = Node(
            edge.child_id,
            edge.parent_id,
            edge.q,
            order,
            edge.batch_id,
        )

    children: dict[int, list[int]] = defaultdict(list)
    for edge in edges:
        children[edge.parent_id].append(edge.child_id)

    selection_events = [
        row
        for row in decisions
        if row.get("event") in {"node_selected", "trajectory_selected"}
    ]
    selection_events.sort(key=lambda row: int(row["batch_id"]))
    completed = {
        int(row["batch_id"]): row
        for row in decisions
        if row.get("event") == "trajectory_verification_completed"
    }
    failures_by_batch: dict[int, int] = defaultdict(int)
    for row in decisions:
        if row.get("event") == "code_generation_failed" and row.get("batch_id") is not None:
            failures_by_batch[int(row["batch_id"])] += 1

    edge_batches = {edge.batch_id for edge in edges}
    observed_batches = set(candidates_by_batch) | edge_batches | set(failures_by_batch)
    if protocol_id == "traceaad-v9.1-trajectory-centered":
        observed_batches &= set(completed)

    prior_results: dict[int, list[dict]] = defaultdict(list)
    events: list[dict] = []
    edges_by_batch: dict[int, list[Edge]] = defaultdict(list)
    for edge in edges:
        edges_by_batch[edge.batch_id].append(edge)

    for selection in selection_events:
        batch_id = int(selection["batch_id"])
        if batch_id not in observed_batches:
            continue
        anchor_id = int(selection.get("selected_node_id"))
        if anchor_id not in nodes:
            continue
        anchor = nodes[anchor_id]
        path = path_to(anchor_id, nodes)
        path_scores = [nodes[item].q for item in path]
        window_ids = path[-WINDOW_NODES:]
        window_scores = [nodes[item].q for item in window_ids]
        deltas = np.diff(window_scores).tolist()
        path_best = max(path_scores)

        existing = [node for node in nodes.values() if node.creation_batch < batch_id]
        if not existing:
            continue
        global_best = max(node.q for node in existing)
        rank = 1 + sum(node.q > anchor.q + EPS for node in existing)
        rank_pct = sum(node.q <= anchor.q + EPS for node in existing) / len(existing)

        previous_direct = [
            edge
            for edge in edges
            if edge.parent_id == anchor_id and edge.batch_id < batch_id
        ]
        direct_deltas = [edge.delta_parent for edge in previous_direct]
        direct_scores = [edge.q for edge in previous_direct]

        previous = prior_results[anchor_id]
        recent = previous[-4:]
        previous_best_outputs = [
            float(item["best_child_q"])
            for item in previous
            if item["best_child_q"] is not None
        ]
        previous_budget_outcomes = [
            (
                float(item["best_child_q"])
                if item["best_child_q"] is not None
                else anchor.q
            )
            for item in previous
        ]
        prev_valid = sum(item["valid_count"] for item in previous)
        prev_requested = sum(item["requested"] for item in previous)

        current_edges = edges_by_batch.get(batch_id, [])
        current_edges = [edge for edge in current_edges if edge.parent_id == anchor_id]
        child_scores = [edge.q for edge in current_edges]
        requested = selection.get("requested_candidates", selection.get("requested_children"))
        if requested is None and batch_id in completed:
            requested = completed[batch_id].get("attempted_candidates")
        requested = max(1, int(requested or len(candidates_by_batch[batch_id]) or 1))
        valid_count = len(child_scores)
        best_child_q = max(child_scores) if child_scores else None
        post_event_best_q = anchor.q if best_child_q is None else max(anchor.q, best_child_q)
        anchor_improved = best_child_q is not None and best_child_q > anchor.q + EPS
        route_advanced = best_child_q is not None and best_child_q > path_best + EPS
        global_advanced = best_child_q is not None and best_child_q > global_best + EPS

        # 新 V9.1 的审计字段必须与无泄漏重算一致。
        if batch_id in completed:
            logged = bool(completed[batch_id].get("route_advanced", False))
            if logged != route_advanced:
                raise AssertionError(
                    f"route label mismatch: {run_dir.name} batch={batch_id}"
                )

        weights = np.arange(1, len(window_scores) + 1, dtype=float)
        weighted_level = float(np.average(window_scores, weights=weights))
        slope = linear_slope(window_scores)
        improve_fraction = (
            sum(delta > EPS for delta in deltas) / len(deltas) if deltas else 0.0
        )
        regress_fraction = (
            sum(delta < -EPS for delta in deltas) / len(deltas) if deltas else 0.0
        )
        direct_improve_fraction = (
            sum(delta > EPS for delta in direct_deltas) / len(direct_deltas)
            if direct_deltas
            else 0.0
        )
        batch_candidates = candidates_by_batch.get(batch_id, [])
        before_orders = [
            int(row["sample_order"])
            for row in batch_candidates
            if row.get("sample_order") is not None
        ]
        sample_before = min(before_orders) - 1 if before_orders else max(
            (node.sample_order for node in existing), default=0
        )

        record = {
            "protocol": protocol,
            "protocol_id": protocol_id,
            "task": task,
            "run": run_dir.name,
            "run_dir": str(run_dir.relative_to(REPO)),
            "status": run_status(run_dir),
            "batch_id": batch_id,
            "sample_before": sample_before,
            "stage_fraction": min(1.0, sample_before / budget),
            "anchor_id": anchor_id,
            "anchor_q": anchor.q,
            "anchor_rank": rank,
            "anchor_rank_pct": rank_pct,
            "anchor_global_gap": anchor.q - global_best,
            "depth": len(path),
            "window_len": len(window_ids),
            "window_level_gap": float(np.mean(window_scores) - anchor.q),
            "window_weighted_level_gap": weighted_level - anchor.q,
            "window_weighted_level_q": weighted_level,
            "window_path_best_gap": anchor.q - path_best,
            "window_path_best_q": path_best,
            "window_net_delta": window_scores[-1] - window_scores[0],
            "window_slope": slope,
            "anchor_plus_window_slope": anchor.q + slope,
            "window_last_delta": deltas[-1] if deltas else 0.0,
            "window_delta_volatility": float(np.std(deltas)) if deltas else 0.0,
            "window_improve_fraction": improve_fraction,
            "window_regress_fraction": regress_fraction,
            "window_turn_rate": sign_change_rate(deltas),
            "anchor_at_path_best": float(anchor.q >= path_best - EPS),
            "direct_child_count": len(previous_direct),
            "direct_best_delta": max(direct_deltas) if direct_deltas else 0.0,
            "direct_mean_delta": float(np.mean(direct_deltas)) if direct_deltas else 0.0,
            "direct_best_global_gap": (
                max(direct_scores) - global_best
                if direct_scores
                else anchor.q - global_best
            ),
            "direct_mean_global_gap": (
                float(np.mean(direct_scores)) - global_best
                if direct_scores
                else anchor.q - global_best
            ),
            "direct_delta_volatility": (
                float(np.std(direct_deltas)) if direct_deltas else 0.0
            ),
            "direct_improve_fraction": direct_improve_fraction,
            "prior_attempt_count": len(previous),
            "prior_route_advance_rate": (
                sum(item["route_advanced"] for item in previous) / len(previous)
                if previous
                else 0.0
            ),
            "prior_recent_advance_rate": (
                sum(item["route_advanced"] for item in recent) / len(recent)
                if recent
                else 0.0
            ),
            "prior_valid_rate": prev_valid / prev_requested if prev_requested else 0.0,
            "prior_mean_child_q": (
                float(np.mean(previous_best_outputs))
                if previous_best_outputs
                else anchor.q
            ),
            "prior_recent_child_q": (
                previous_best_outputs[-1] if previous_best_outputs else anchor.q
            ),
            "prior_max_child_q": (
                max(previous_best_outputs) if previous_best_outputs else anchor.q
            ),
            "anchor_child_blend_q": (
                (anchor.q + sum(previous_best_outputs))
                / (1 + len(previous_best_outputs))
            ),
            "anchor_event_blend_q": (
                (anchor.q + sum(previous_budget_outcomes))
                / (1 + len(previous_budget_outcomes))
            ),
            "requested": requested,
            "valid_count": valid_count,
            "valid_fraction": valid_count / requested,
            "best_child_q": best_child_q,
            "best_generated_global_margin": (
                None if best_child_q is None else best_child_q - global_best
            ),
            "global_gain": (
                0.0 if best_child_q is None else max(0.0, best_child_q - global_best)
            ),
            "post_event_best_q": post_event_best_q,
            "immediate_gain": post_event_best_q - anchor.q,
            "anchor_improved": int(anchor_improved),
            "route_advanced": int(route_advanced),
            "global_advanced": int(global_advanced),
            "_child_ids": [edge.child_id for edge in current_edges],
            "_path_best": path_best,
        }
        events.append(record)
        prior_results[anchor_id].append(record)

    # 延迟价值只在这次生成的真实子树中观察。没有后续选择的项视为删失，不记失败。
    selections_by_node: dict[int, list[int]] = defaultdict(list)
    for selection in selection_events:
        if selection.get("selected_node_id") is not None:
            selections_by_node[int(selection["selected_node_id"])].append(
                int(selection["batch_id"])
            )

    for event in events:
        queue = deque((child_id, 1) for child_id in event["_child_ids"])
        visited: set[int] = set()
        delayed_scores: list[float] = []
        followed = False
        while queue:
            node_id, distance = queue.popleft()
            if node_id in visited or distance > DELAYED_DEPTH or node_id not in nodes:
                continue
            visited.add(node_id)
            delayed_scores.append(nodes[node_id].q)
            if any(batch > event["batch_id"] for batch in selections_by_node[node_id]):
                followed = True
            if distance < DELAYED_DEPTH:
                queue.extend((child, distance + 1) for child in children.get(node_id, ()))
        delayed_best = max([event["anchor_q"], *delayed_scores])
        event["followup_observed"] = int(followed)
        event["delayed_best_q_d3"] = delayed_best
        event["delayed_route_advanced_d3"] = int(
            followed and delayed_best > event["_path_best"] + EPS
        )
        event["delayed_rescue_d3"] = int(
            followed
            and not event["route_advanced"]
            and delayed_best > event["_path_best"] + EPS
        )

    info = {
        "protocol": protocol,
        "protocol_id": protocol_id,
        "task": task,
        "run": run_dir.name,
        "status": run_status(run_dir),
        "candidate_rows": len(candidates),
        "edge_rows": len(edges),
        "selection_rows": len(selection_events),
        "analyzed_events": len(events),
        "skipped_partial_jsonl_rows": (
            skipped_candidates + skipped_decisions + skipped_edges
        ),
    }
    return events, info


FEATURES = [
    "anchor_q",
    "anchor_rank_pct",
    "anchor_global_gap",
    "window_path_best_gap",
    "window_level_gap",
    "window_weighted_level_gap",
    "window_net_delta",
    "window_slope",
    "window_last_delta",
    "window_delta_volatility",
    "window_improve_fraction",
    "window_turn_rate",
    "anchor_at_path_best",
    "direct_child_count",
    "direct_best_delta",
    "direct_mean_delta",
    "direct_best_global_gap",
    "direct_mean_global_gap",
    "direct_improve_fraction",
    "prior_attempt_count",
    "prior_route_advance_rate",
    "prior_recent_advance_rate",
    "prior_valid_rate",
]

HEURISTIC_SCORES = [
    "anchor_q",
    "window_path_best_q",
    "window_weighted_level_q",
    "anchor_plus_window_slope",
    "prior_mean_child_q",
    "prior_recent_child_q",
    "prior_max_child_q",
    "anchor_child_blend_q",
    "anchor_event_blend_q",
]

MODELS = {
    "anchor": ["anchor_q", "anchor_global_gap"],
    "anchor_shape": [
        "anchor_q",
        "anchor_global_gap",
        "window_path_best_gap",
        "window_level_gap",
        "window_slope",
        "window_delta_volatility",
        "window_improve_fraction",
        "window_turn_rate",
        "window_len",
    ],
    "anchor_local": [
        "anchor_q",
        "anchor_global_gap",
        "window_path_best_gap",
        "window_level_gap",
        "window_slope",
        "window_delta_volatility",
        "window_improve_fraction",
        "window_turn_rate",
        "window_len",
        "direct_child_count",
        "direct_best_delta",
        "direct_mean_delta",
        "direct_best_global_gap",
        "direct_mean_global_gap",
        "direct_improve_fraction",
    ],
    "anchor_empirical": [
        "anchor_q",
        "anchor_global_gap",
        "prior_attempt_count",
        "prior_route_advance_rate",
        "prior_recent_advance_rate",
        "prior_valid_rate",
    ],
    "anchor_evidence": [
        "anchor_q",
        "anchor_global_gap",
        "direct_child_count",
        "direct_best_delta",
        "direct_mean_delta",
        "direct_best_global_gap",
        "direct_mean_global_gap",
        "direct_improve_fraction",
        "prior_attempt_count",
        "prior_route_advance_rate",
        "prior_recent_advance_rate",
        "prior_valid_rate",
    ],
    "full": [
        "anchor_q",
        "anchor_global_gap",
        "window_path_best_gap",
        "window_level_gap",
        "window_slope",
        "window_delta_volatility",
        "window_improve_fraction",
        "window_turn_rate",
        "window_len",
        "direct_child_count",
        "direct_best_delta",
        "direct_mean_delta",
        "direct_best_global_gap",
        "direct_mean_global_gap",
        "direct_improve_fraction",
        "prior_attempt_count",
        "prior_route_advance_rate",
        "prior_recent_advance_rate",
        "prior_valid_rate",
    ],
}


def describe(events: pd.DataFrame, run_info: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for protocol, frame in events.groupby("protocol", sort=False):
        followed = frame[frame["followup_observed"] == 1]
        rows.append(
            {
                "protocol": protocol,
                "runs": int(frame["run"].nunique()),
                "events": int(len(frame)),
                "route_advance_rate": float(frame["route_advanced"].mean()),
                "global_advance_rate": float(frame["global_advanced"].mean()),
                "route_to_global_conversion": (
                    float(frame["global_advanced"].sum() / frame["route_advanced"].sum())
                    if frame["route_advanced"].sum()
                    else None
                ),
                "anchor_improve_rate": float(frame["anchor_improved"].mean()),
                "valid_fraction": float(frame["valid_fraction"].mean()),
                "median_depth": float(frame["depth"].median()),
                "window_len_ge4": float((frame["window_len"] >= 4).mean()),
                "window_len_eq8": float((frame["window_len"] == 8).mean()),
                "has_direct_evidence": float((frame["direct_child_count"] > 0).mean()),
                "multiple_direct_branches": float(
                    (frame["direct_child_count"] >= 2).mean()
                ),
                "reselected_anchor": float((frame["prior_attempt_count"] > 0).mean()),
                "followup_events": int(len(followed)),
                "delayed_rescue_rate_given_followup": (
                    float(followed["delayed_rescue_d3"].mean())
                    if len(followed)
                    else None
                ),
            }
        )
    return rows


def rank_bands(events: pd.DataFrame) -> list[dict]:
    """描述预算落在哪个锚点质量区域；不把观察频率解释成反事实收益。"""
    table = events.copy()
    table["rank_band"] = pd.cut(
        table["anchor_rank"],
        bins=[0, 1, 3, 10, np.inf],
        labels=["1", "2-3", "4-10", ">10"],
        right=True,
    )
    result: list[dict] = []
    for protocol, protocol_frame in table.groupby("protocol", observed=True):
        total = len(protocol_frame)
        for band, frame in protocol_frame.groupby("rank_band", observed=True):
            result.append(
                {
                    "protocol": protocol,
                    "rank_band": str(band),
                    "events": int(len(frame)),
                    "budget_share": float(len(frame) / total),
                    "route_advance_rate": float(frame["route_advanced"].mean()),
                    "global_advance_rate": float(frame["global_advanced"].mean()),
                    "valid_fraction": float(frame["valid_fraction"].mean()),
                }
            )
    return result


def associations(events: pd.DataFrame, *, target: str) -> list[dict]:
    """先逐 run 计算 Spearman，再汇总，避免大 run 支配结论。"""
    per_run: list[dict] = []
    for (protocol, run), frame in events.groupby(["protocol", "run"]):
        if len(frame) < 20 or frame[target].nunique() < 2:
            continue
        for feature in FEATURES:
            observed = frame[[feature, target]].dropna()
            if observed[feature].nunique() < 2 or observed[target].nunique() < 2:
                continue
            rho = stats.spearmanr(observed[feature], observed[target]).statistic
            if finite_number(rho):
                per_run.append(
                    {
                        "target": target,
                        "protocol": protocol,
                        "run": run,
                        "feature": feature,
                        "rho": float(rho),
                    }
                )
    if not per_run:
        return []
    result = []
    table = pd.DataFrame(per_run)
    for (protocol, feature), frame in table.groupby(["protocol", "feature"]):
        values = frame["rho"].to_numpy()
        result.append(
            {
                "protocol": protocol,
                "target": target,
                "feature": feature,
                "runs": int(len(values)),
                "median_rho": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "positive_run_fraction": float((values > 0).mean()),
            }
        )
    return result


def cross_validated_models(events: pd.DataFrame) -> list[dict]:
    folds: list[dict] = []
    for (protocol, task), frame in events.groupby(["protocol", "task"]):
        runs = sorted(frame["run"].unique())
        if len(runs) < 2:
            continue
        for held_run in runs:
            train = frame[frame["run"] != held_run]
            test = frame[frame["run"] == held_run]
            if (
                len(train) < 40
                or len(test) < 20
                or train["route_advanced"].nunique() < 2
                or test["route_advanced"].nunique() < 2
            ):
                continue
            y_train = train["route_advanced"].to_numpy()
            y_test = test["route_advanced"].to_numpy()
            global_train = train["global_advanced"].to_numpy()
            global_test = test["global_advanced"].to_numpy()
            for model_name, features in MODELS.items():
                classifier = make_pipeline(
                    SimpleImputer(strategy="constant", fill_value=0.0),
                    StandardScaler(),
                    LogisticRegression(C=0.25, max_iter=2000),
                )
                classifier.fit(train[features], y_train)
                probability = classifier.predict_proba(test[features])[:, 1]
                global_auc = None
                global_ap = None
                if len(np.unique(global_train)) == 2 and len(np.unique(global_test)) == 2:
                    global_classifier = make_pipeline(
                        SimpleImputer(strategy="constant", fill_value=0.0),
                        StandardScaler(),
                        LogisticRegression(C=0.25, max_iter=2000),
                    )
                    global_classifier.fit(train[features], global_train)
                    global_probability = global_classifier.predict_proba(test[features])[:, 1]
                    global_auc = float(roc_auc_score(global_test, global_probability))
                    global_ap = float(average_precision_score(global_test, global_probability))

                regressor = make_pipeline(
                    SimpleImputer(strategy="constant", fill_value=0.0),
                    StandardScaler(),
                    Ridge(alpha=10.0),
                )
                regressor.fit(train[features], train["post_event_best_q"])
                predicted_q = regressor.predict(test[features])
                q_rho = stats.spearmanr(predicted_q, test["post_event_best_q"]).statistic

                valid_train = train[train["best_child_q"].notna()]
                valid_test = test[test["best_child_q"].notna()]
                child_q_mae = None
                child_q_rho = None
                if len(valid_train) >= 30 and len(valid_test) >= 15:
                    child_regressor = make_pipeline(
                        SimpleImputer(strategy="constant", fill_value=0.0),
                        StandardScaler(),
                        Ridge(alpha=10.0),
                    )
                    child_regressor.fit(valid_train[features], valid_train["best_child_q"])
                    predicted_child_q = child_regressor.predict(valid_test[features])
                    child_q_mae = float(
                        mean_absolute_error(valid_test["best_child_q"], predicted_child_q)
                    )
                    child_q_rho = float(
                        stats.spearmanr(
                            predicted_child_q, valid_test["best_child_q"]
                        ).statistic
                    )
                folds.append(
                    {
                        "protocol": protocol,
                        "task": task,
                        "held_run": held_run,
                        "model": model_name,
                        "n_test": int(len(test)),
                        "route_auc": float(roc_auc_score(y_test, probability)),
                        "route_ap": float(average_precision_score(y_test, probability)),
                        "global_auc": global_auc,
                        "global_ap": global_ap,
                        "post_q_mae": float(
                            mean_absolute_error(test["post_event_best_q"], predicted_q)
                        ),
                        "post_q_spearman": float(q_rho),
                        "child_q_mae": child_q_mae,
                        "child_q_spearman": child_q_rho,
                    }
                )
            baseline_q = test["anchor_q"].to_numpy()
            identity_global_auc = None
            identity_global_ap = None
            if len(np.unique(global_test)) == 2:
                identity_global_auc = float(roc_auc_score(global_test, baseline_q))
                identity_global_ap = float(average_precision_score(global_test, baseline_q))
            folds.append(
                {
                    "protocol": protocol,
                    "task": task,
                    "held_run": held_run,
                    "model": "identity_anchor_score",
                    "n_test": int(len(test)),
                    "route_auc": float(roc_auc_score(y_test, baseline_q)),
                    "route_ap": float(average_precision_score(y_test, baseline_q)),
                    "global_auc": identity_global_auc,
                    "global_ap": identity_global_ap,
                    "post_q_mae": float(
                        mean_absolute_error(test["post_event_best_q"], baseline_q)
                    ),
                    "post_q_spearman": float(
                        stats.spearmanr(baseline_q, test["post_event_best_q"]).statistic
                    ),
                    "child_q_mae": (
                        None
                        if test["best_child_q"].dropna().empty
                        else float(
                            mean_absolute_error(
                                test.loc[test["best_child_q"].notna(), "best_child_q"],
                                test.loc[test["best_child_q"].notna(), "anchor_q"],
                            )
                        )
                    ),
                    "child_q_spearman": (
                        None
                        if test["best_child_q"].dropna().nunique() < 2
                        else float(
                            stats.spearmanr(
                                test.loc[test["best_child_q"].notna(), "anchor_q"],
                                test.loc[test["best_child_q"].notna(), "best_child_q"],
                            ).statistic
                        )
                    ),
                }
            )
    return folds


def aggregate_cv(folds: list[dict]) -> list[dict]:
    if not folds:
        return []
    frame = pd.DataFrame(folds)
    result = []
    for (protocol, model), group in frame.groupby(["protocol", "model"]):
        result.append(
            {
                "protocol": protocol,
                "model": model,
                "folds": int(len(group)),
                "route_auc_mean": float(group["route_auc"].mean()),
                "route_auc_median": float(group["route_auc"].median()),
                "route_ap_mean": float(group["route_ap"].mean()),
                "global_auc_mean": (
                    None
                    if group["global_auc"].dropna().empty
                    else float(group["global_auc"].mean())
                ),
                "global_ap_mean": (
                    None
                    if group["global_ap"].dropna().empty
                    else float(group["global_ap"].mean())
                ),
                "post_q_mae_mean": float(group["post_q_mae"].mean()),
                "post_q_spearman_mean": float(group["post_q_spearman"].mean()),
                "child_q_mae_mean": (
                    None
                    if group["child_q_mae"].dropna().empty
                    else float(group["child_q_mae"].mean())
                ),
                "child_q_spearman_mean": (
                    None
                    if group["child_q_spearman"].dropna().empty
                    else float(group["child_q_spearman"].mean())
                ),
            }
        )
    return result


def heuristic_scores(events: pd.DataFrame) -> list[dict]:
    """直接检验无需拟合权重、可原样实现的候选分数。"""
    per_run: list[dict] = []
    for (protocol, run), frame in events.groupby(["protocol", "run"]):
        for score in HEURISTIC_SCORES:
            row = {"protocol": protocol, "run": run, "score": score}
            if frame["route_advanced"].nunique() == 2:
                row["route_auc"] = float(
                    roc_auc_score(frame["route_advanced"], frame[score])
                )
            else:
                row["route_auc"] = None
            if frame["global_advanced"].nunique() == 2:
                row["global_auc"] = float(
                    roc_auc_score(frame["global_advanced"], frame[score])
                )
            else:
                row["global_auc"] = None
            valid = frame[frame["best_child_q"].notna()]
            if (
                len(valid) >= 15
                and valid["best_child_q"].nunique() >= 2
                and valid[score].nunique() >= 2
            ):
                row["child_q_spearman"] = float(
                    stats.spearmanr(valid[score], valid["best_child_q"]).statistic
                )
                row["global_margin_spearman"] = float(
                    stats.spearmanr(
                        valid[score], valid["best_generated_global_margin"]
                    ).statistic
                )
            else:
                row["child_q_spearman"] = None
                row["global_margin_spearman"] = None
            per_run.append(row)

    table = pd.DataFrame(per_run)
    result: list[dict] = []
    for (protocol, score), frame in table.groupby(["protocol", "score"]):
        item = {"protocol": protocol, "score": score, "runs": int(len(frame))}
        for metric in (
            "route_auc",
            "global_auc",
            "child_q_spearman",
            "global_margin_spearman",
        ):
            values = frame[metric].dropna()
            item[f"{metric}_median"] = (
                None if values.empty else float(values.median())
            )
            item[f"{metric}_positive_fraction"] = (
                None
                if values.empty
                else float((values > (0.5 if "auc" in metric else 0.0)).mean())
            )
        result.append(item)
    return result


def fmt_pct(value) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def print_report(payload: dict) -> None:
    print(f"快照时间: {payload['snapshot_time']}")
    print(
        f"run={len(payload['runs'])}, event={payload['n_events']}, "
        f"忽略末尾不完整 JSONL={payload['skipped_partial_jsonl_rows']}\n"
    )
    print("=== 1. 事件覆盖与直接结果 ===")
    for row in payload["descriptive"]:
        print(
            f"{row['protocol']:<18} runs={row['runs']:>2} events={row['events']:>5} "
            f"route={fmt_pct(row['route_advance_rate'])} "
            f"global={fmt_pct(row['global_advance_rate'])} "
            f"global/route={fmt_pct(row['route_to_global_conversion'])} "
            f"valid={fmt_pct(row['valid_fraction'])} depth_med={row['median_depth']:.1f} "
            f"W>=4={fmt_pct(row['window_len_ge4'])} W=8={fmt_pct(row['window_len_eq8'])} "
            f"child>0={fmt_pct(row['has_direct_evidence'])} child>=2={fmt_pct(row['multiple_direct_branches'])} "
            f"delayed_rescue={fmt_pct(row['delayed_rescue_rate_given_followup'])}"
        )

    print("\n=== 2. 锚点当时排名区间 ===")
    rank = pd.DataFrame(payload["rank_bands"])
    if not rank.empty:
        for protocol, frame in rank.groupby("protocol", sort=False):
            print(f"[{protocol}]")
            for _, row in frame.iterrows():
                print(
                    f"  rank={row['rank_band']:<4} events={int(row['events']):>5} "
                    f"budget={fmt_pct(row['budget_share'])} "
                    f"route={fmt_pct(row['route_advance_rate'])} "
                    f"global={fmt_pct(row['global_advance_rate'])}"
                )

    print("\n=== 3. 与路线推进 / 全局推进的逐 run Spearman（按 |中位数| 前 8） ===")
    assoc = pd.DataFrame(payload["associations"])
    if not assoc.empty:
        for (target, protocol), frame in assoc.groupby(["target", "protocol"], sort=False):
            print(f"[{target} / {protocol}]")
            for _, row in frame.assign(abs_rho=frame["median_rho"].abs()).nlargest(
                8, "abs_rho"
            ).iterrows():
                print(
                    f"  {row['feature']:<30} rho_med={row['median_rho']:+.3f} "
                    f"IQR=[{row['q25']:+.3f},{row['q75']:+.3f}] "
                    f"positive={row['positive_run_fraction']:.0%} n={int(row['runs'])}"
                )

    print("\n=== 4. 按 task 内留一 run 的预测检验 ===")
    cv = pd.DataFrame(payload["cv_aggregate"])
    if not cv.empty:
        for protocol, frame in cv.groupby("protocol", sort=False):
            print(f"[{protocol}]")
            for _, row in frame.sort_values("route_auc_mean", ascending=False).iterrows():
                print(
                    f"  {row['model']:<24} folds={int(row['folds']):>2} "
                    f"AUC={row['route_auc_mean']:.3f} AP={row['route_ap_mean']:.3f} "
                    f"globalAUC={row['global_auc_mean'] if row['global_auc_mean'] is not None else float('nan'):.3f} "
                    f"childQ_rho={row['child_q_spearman_mean'] if row['child_q_spearman_mean'] is not None else float('nan'):.3f} "
                    f"postQ_rho={row['post_q_spearman_mean']:.3f} "
                    f"MAE={row['post_q_mae_mean']:.4g}"
                )

    print("\n=== 5. 无拟合候选分数的逐 run 中位表现 ===")
    heuristic = pd.DataFrame(payload["heuristic_scores"])
    if not heuristic.empty:
        for protocol, frame in heuristic.groupby("protocol", sort=False):
            print(f"[{protocol}]")
            for _, row in frame.sort_values(
                "global_margin_spearman_median", ascending=False
            ).iterrows():
                print(
                    f"  {row['score']:<28} "
                    f"globalAUC={row['global_auc_median'] if row['global_auc_median'] is not None else float('nan'):.3f} "
                    f"childQ_rho={row['child_q_spearman_median'] if row['child_q_spearman_median'] is not None else float('nan'):.3f} "
                    f"margin_rho={row['global_margin_spearman_median'] if row['global_margin_spearman_median'] is not None else float('nan'):.3f}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    all_events: list[dict] = []
    run_info: list[dict] = []
    for run_dir in discover_runs():
        events, info = parse_run(run_dir)
        all_events.extend(events)
        run_info.append(info)
    if not all_events:
        raise SystemExit("没有找到可分析的 V9 / V9.1 预算事件")

    frame = pd.DataFrame(all_events)
    assoc = associations(frame, target="route_advanced")
    assoc += associations(frame, target="global_advanced")
    assoc += associations(frame, target="best_generated_global_margin")
    folds = cross_validated_models(frame)
    payload = {
        "snapshot_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "window_nodes": WINDOW_NODES,
        "delayed_depth": DELAYED_DEPTH,
        "n_events": len(frame),
        "skipped_partial_jsonl_rows": sum(
            item["skipped_partial_jsonl_rows"] for item in run_info
        ),
        "runs": run_info,
        "descriptive": describe(frame, run_info),
        "rank_bands": rank_bands(frame),
        "associations": assoc,
        "cv_folds": folds,
        "cv_aggregate": aggregate_cv(folds),
        "heuristic_scores": heuristic_scores(frame),
        "limitations": [
            "只观察历史策略实际选择的锚点，没有未选择锚点的反事实结果。",
            "旧协议和新协议分别分析；跨协议相同符号只算稳健性线索。",
            "延迟价值只对后来真实获得扩展的子树可见，存在选择性随访。",
            "同一锚点的不同窗口过去没有随机对照，不能由这些日志确定最佳窗口。",
            "全部统计是探索性分析，评分公式仍需独立 V9.2 对照验证。",
        ],
    }
    print_report(payload)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nJSON: {args.json_out}")


if __name__ == "__main__":
    main()

"""生存管理中路线差异 reserve 使用的轻量相似度。"""

from __future__ import annotations

import re

from .derivation_graph import DerivationGraph
from .schema import Trajectory

_CODE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def normalize_code(code: str) -> str:
    code = re.sub(r"#.*", "", code)
    return re.sub(r"\s+", " ", code).strip()


def code_tokens(code: str) -> frozenset[str]:
    return frozenset(_CODE_TOKEN_RE.findall(normalize_code(code)))


def code_similarity(code_a: str, code_b: str) -> float:
    ta, tb = code_tokens(code_a), code_tokens(code_b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def trajectory_pattern(
    graph: DerivationGraph, trajectory: Trajectory
) -> frozenset[str]:
    """(operator, outcome) 对的集合，刻画轨迹的搜索行为指纹。"""
    pairs: set[str] = set()
    for eid in trajectory.edge_ids:
        edge = graph.get_edge(eid)
        pairs.add(f"{edge.operator}|{edge.outcome}")
    return frozenset(pairs)


def trajectory_pattern_similarity(
    pat_a: frozenset[str], pat_b: frozenset[str]
) -> float:
    if not pat_a and not pat_b:
        return 1.0
    if not pat_a or not pat_b:
        return 0.0
    return len(pat_a & pat_b) / len(pat_a | pat_b)


def trajectory_similarity(
    *,
    graph: DerivationGraph,
    left: Trajectory,
    right: Trajectory,
) -> float:
    """Return a lightweight code, idea and route similarity for diversity sampling."""
    if left.id == right.id:
        return 1.0
    w_code, w_idea, w_traj = (0.5, 0.3, 0.2)
    code = code_similarity(
        graph.get_node(left.endpoint_id).code,
        graph.get_node(right.endpoint_id).code,
    )
    idea = _jaccard(
        trajectory_idea_tokens(graph, left),
        trajectory_idea_tokens(graph, right),
    )
    pattern = trajectory_pattern_similarity(
        trajectory_pattern(graph, left),
        trajectory_pattern(graph, right),
    )
    return w_code * code + w_idea * idea + w_traj * pattern


def trajectory_idea_tokens(
    graph: DerivationGraph,
    trajectory: Trajectory,
) -> frozenset[str]:
    """Return a cheap lexical signature of the ideas and actions on a route."""
    texts = [graph.get_node(node_id).idea for node_id in trajectory.node_ids]
    texts.extend(graph.get_edge(edge_id).action for edge_id in trajectory.edge_ids)
    return frozenset(
        token.lower()
        for text in texts
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text or "")
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

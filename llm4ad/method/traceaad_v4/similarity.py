"""生存管理中路线差异 reserve 使用的轻量相似度。"""

from __future__ import annotations

import re

from .derivation_graph import DerivationGraph
from .schema import Trajectory

_CODE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_IDEA_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _normalize_code(code: str) -> str:
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"```python|```", "", code)
    return re.sub(r"\s+", " ", code).strip()


def code_similarity(code_a: str, code_b: str) -> float:
    ta = frozenset(_CODE_TOKEN_RE.findall(_normalize_code(code_a)))
    tb = frozenset(_CODE_TOKEN_RE.findall(_normalize_code(code_b)))
    return _jaccard(ta, tb)


def trajectory_pattern(
    graph: DerivationGraph, trajectory: Trajectory
) -> frozenset[str]:
    """(operator, outcome) pairs as a lightweight search-behavior fingerprint."""
    return frozenset(
        f"{graph.get_edge(eid).operator}|{graph.get_edge(eid).outcome}"
        for eid in trajectory.edge_ids
    )


def trajectory_idea_tokens(
    graph: DerivationGraph,
    trajectory: Trajectory,
) -> frozenset[str]:
    texts = [graph.get_node(node_id).idea for node_id in trajectory.node_ids]
    texts.extend(graph.get_edge(edge_id).action for edge_id in trajectory.edge_ids)
    return frozenset(
        token.lower()
        for text in texts
        for token in _IDEA_TOKEN_RE.findall(text or "")
    )


def trajectory_similarity(
    *,
    graph: DerivationGraph,
    left: Trajectory,
    right: Trajectory,
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> float:
    """Code / idea / route-pattern similarity for diversity reserve."""
    if left.id == right.id:
        return 1.0
    w_code, w_idea, w_traj = weights
    total = w_code + w_idea + w_traj
    if total <= 0:
        return 0.0
    code = code_similarity(
        graph.get_node(left.endpoint_id).code,
        graph.get_node(right.endpoint_id).code,
    )
    idea = _jaccard(
        trajectory_idea_tokens(graph, left),
        trajectory_idea_tokens(graph, right),
    )
    pattern = _jaccard(
        trajectory_pattern(graph, left),
        trajectory_pattern(graph, right),
    )
    return (w_code * code + w_idea * idea + w_traj * pattern) / total

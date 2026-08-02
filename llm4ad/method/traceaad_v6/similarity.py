"""Lightweight trajectory similarity helpers retained for analysis."""

from __future__ import annotations

import re

from .derivation_graph import DerivationGraph
from .schema import Trajectory

_CODE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WORD_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]+")

DEFAULT_SIMILARITY_WEIGHTS = (0.5, 0.3, 0.2)


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
    pairs: set[str] = set()
    for eid in trajectory.edge_ids:
        edge = graph.get_edge(eid)
        pairs.add(f"{edge.operator}|{edge.outcome}")
    return frozenset(pairs)


def trajectory_pattern_similarity(
    pat_a: frozenset[str], pat_b: frozenset[str]
) -> float:
    return _jaccard(pat_a, pat_b)


def trajectory_idea_tokens(
    graph: DerivationGraph,
    trajectory: Trajectory,
) -> frozenset[str]:
    """Action and Implemented Idea tokens on the retained path."""
    texts: list[str] = []
    for edge_id in trajectory.edge_ids:
        edge = graph.get_edge(edge_id)
        texts.append(edge.action)
        texts.append(graph.get_node(edge.child_id).idea)
    return frozenset(
        token.lower() for text in texts for token in _WORD_TOKEN_RE.findall(text or "")
    )


def trajectory_similarity(
    *,
    graph: DerivationGraph,
    left: Trajectory,
    right: Trajectory,
    weights: tuple[float, float, float] = DEFAULT_SIMILARITY_WEIGHTS,
) -> float:
    """Sim = 0.5 S_code + 0.3 S_idea + 0.2 S_path over compact-best codes."""
    if left.id == right.id:
        return 1.0
    w_code, w_idea, w_path = weights
    code = code_similarity(
        graph.get_node(left.compact_best_id).code,
        graph.get_node(right.compact_best_id).code,
    )
    idea = _jaccard(
        trajectory_idea_tokens(graph, left),
        trajectory_idea_tokens(graph, right),
    )
    pattern = trajectory_pattern_similarity(
        trajectory_pattern(graph, left),
        trajectory_pattern(graph, right),
    )
    return w_code * code + w_idea * idea + w_path * pattern


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


__all__ = [
    "DEFAULT_SIMILARITY_WEIGHTS",
    "code_similarity",
    "code_tokens",
    "normalize_code",
    "trajectory_idea_tokens",
    "trajectory_pattern",
    "trajectory_pattern_similarity",
    "trajectory_similarity",
]

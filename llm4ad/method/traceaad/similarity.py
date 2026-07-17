"""多层相似度：程序层 / 机制层 / 轨迹层。

不引入外部 embedding 模型：code 层用规范化 token Jaccard，机制层用 mechanism_tag 集合
Jaccard，轨迹层用 (operator,outcome) 序列 Jaccard。组合相似度供 novelty gate 与
V_diversity / V_novelty 共用（design §7.2/§8）。
"""
from __future__ import annotations

import re

from .derivation_graph import DerivationGraph
from .schema import Trajectory

_CODE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def normalize_code(code: str) -> str:
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"```python|```", "", code)
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


def mechanism_profile(graph: DerivationGraph, trajectory: Trajectory) -> frozenset[str]:
    tags = {graph.get_node(trajectory.base_id).mechanism_tag}
    for eid in trajectory.edge_ids:
        tags.add(graph.get_edge(eid).mechanism_tag)
    return frozenset(tags)


def mechanism_similarity(profile_a: frozenset[str], profile_b: frozenset[str]) -> float:
    if not profile_a and not profile_b:
        return 1.0
    if not profile_a or not profile_b:
        return 0.0
    return len(profile_a & profile_b) / len(profile_a | profile_b)


def trajectory_pattern(graph: DerivationGraph, trajectory: Trajectory) -> frozenset[str]:
    """(operator, outcome) 对的集合，刻画轨迹的搜索行为指纹。"""
    pairs: set[str] = set()
    for eid in trajectory.edge_ids:
        edge = graph.get_edge(eid)
        pairs.add(f"{edge.operator}|{edge.outcome}")
    return frozenset(pairs)


def trajectory_pattern_similarity(pat_a: frozenset[str], pat_b: frozenset[str]) -> float:
    if not pat_a and not pat_b:
        return 1.0
    if not pat_a or not pat_b:
        return 0.0
    return len(pat_a & pat_b) / len(pat_a | pat_b)


def max_similarity_to_active(
    *,
    graph: DerivationGraph,
    candidate: Trajectory,
    others: tuple[Trajectory, ...],
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> float:
    """candidate 与一组活跃 trajectory 的最大相似度（novelty gate 用）。"""
    if not others:
        return 0.0
    w_code, w_mech, w_traj = weights
    best = 0.0
    cand_code_tokens = code_tokens(graph.get_node(candidate.endpoint_id).code)
    cand_profile = mechanism_profile(graph, candidate)
    cand_pattern = trajectory_pattern(graph, candidate)
    for other in others:
        if other.id == candidate.id:
            continue
        sim_code = _jaccard(cand_code_tokens, code_tokens(graph.get_node(other.endpoint_id).code))
        sim_mech = mechanism_similarity(cand_profile, mechanism_profile(graph, other))
        sim_pat = trajectory_pattern_similarity(cand_pattern, trajectory_pattern(graph, other))
        total = w_code + w_mech + w_traj
        sim = (w_code * sim_code + w_mech * sim_mech + w_traj * sim_pat) / total if total > 0 else 0.0
        if sim > best:
            best = sim
    return best


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

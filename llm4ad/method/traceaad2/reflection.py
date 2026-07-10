"""蒸馏回路 + 反思回路（design §8）。

- distill：周期性从 TrajectoryMemory 提炼机制模式到 PatternMemory（跨轨迹泛化证据）。
- reflect：触发式对比 best vs worst，产出 lesson / anti_pattern，注入 context §6.C。
"""
from __future__ import annotations

from .derivation_graph import DerivationGraph
from .feedback import RankingModel
from .pattern_memory import PatternMemory
from .trajectory_memory import TrajectoryMemory


def distill(
    *,
    memory: TrajectoryMemory,
    graph: DerivationGraph,
    pattern_memory: PatternMemory,
    maximize: bool,
    iteration: int,
    min_support: int = 2,
) -> int:
    """Distill each graph edge exactly once into mechanism and operator credit."""
    stats: dict[str, dict] = {}
    conditioned_stats: dict[tuple[str, str], dict] = {}
    for edge in graph.edges():
        improved = edge.delta is not None and edge.delta > 0
        pattern_memory.record_mechanism_outcome(
            operator=edge.operator,
            mechanism_tag=edge.mechanism_tag,
            support_id=edge.id,
            success=improved,
            iteration=edge.iteration if edge.iteration is not None else iteration,
        )
        st = stats.setdefault(edge.mechanism_tag, {"edge_ids": [], "improved": 0})
        st["edge_ids"].append(edge.id)
        if improved:
            st["improved"] += 1
        conditioned = conditioned_stats.setdefault(
            (edge.operator, edge.mechanism_tag),
            {"edge_ids": [], "improved": 0},
        )
        conditioned["edge_ids"].append(edge.id)
        if improved:
            conditioned["improved"] += 1
    added = 0
    for tag, st in stats.items():
        n_attempts = len(st["edge_ids"])
        gen_score = st["improved"] / n_attempts
        if n_attempts >= min_support and gen_score >= 0.4:
            if pattern_memory.clear_anti_pattern(tag):
                added += 1
        if n_attempts >= min_support and st["improved"] > 0:
            for edge_id in st["edge_ids"]:
                pattern_memory.upsert_mechanism(
                    mechanism_tag=tag,
                    text=(f"Mechanism '{tag}' improved fitness in "
                          f"{st['improved']}/{n_attempts} unique graph edges."),
                    generalization_score=gen_score,
                    support_id=edge_id,
                    updated_iter=iteration,
                )
            added += 1
    for (operator, tag), st in conditioned_stats.items():
        n_attempts = len(st["edge_ids"])
        if n_attempts < max(min_support, 5):
            continue
        gen_score = st["improved"] / n_attempts
        if gen_score < 0.2:
            existed = pattern_memory.is_anti_pattern(tag, operator=operator)
            pattern_memory.add(
                kind="anti_pattern",
                text=(f"Mechanism '{tag}' under operator '{operator}' improved only "
                      f"{gen_score:.0%} of {n_attempts} unique graph edges; deprioritize "
                      "in this operator context."),
                mechanism_tag=tag,
                support_ids=tuple(st["edge_ids"]),
                generalization_score=gen_score,
                confidence=0.8,
                updated_iter=iteration,
                operator=operator,
            )
            if not existed:
                added += 1
        elif pattern_memory.clear_anti_pattern(tag, operator=operator):
            added += 1
    return added


def reflect(
    *,
    memory: TrajectoryMemory,
    graph: DerivationGraph,
    pattern_memory: PatternMemory,
    ranking: RankingModel,
    maximize: bool,
    iteration: int,
    window: int = 20,
) -> dict | None:
    """best-vs-worst 对比 → lesson / anti_pattern。返回对比快照（供日志）。"""
    contrast = ranking.contrast(graph=graph, memory=memory, maximize=maximize, window=window)
    if contrast is None:
        return None
    best, worst = contrast["best"], contrast["worst"]
    if best["mechanism_tag"] != "other":
        pattern_memory.add(
            kind="lesson",
            text=(f"Mechanism '{best['mechanism_tag']}' ranks strongly in current pairwise "
                  "comparisons; prefer evidence-backed variants."),
            mechanism_tag=best["mechanism_tag"],
            support_ids=(best["node_id"],),
            generalization_score=0.6,
            confidence=0.7,
            updated_iter=iteration,
        )
    if worst["mechanism_tag"] != "other" and worst["mechanism_tag"] != best["mechanism_tag"]:
        pattern_memory.add(
            kind="anti_pattern",
            text=(f"Mechanism '{worst['mechanism_tag']}' tends to underperform; treat as low priority."),
            mechanism_tag=worst["mechanism_tag"],
            support_ids=(worst["node_id"],),
            generalization_score=0.2,
            confidence=0.6,
            updated_iter=iteration,
        )
    return contrast

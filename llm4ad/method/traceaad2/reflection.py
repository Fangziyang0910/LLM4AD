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
    """统计每个 mechanism 在所有 active trajectory 边上的改进表现，沉淀为机制模式。"""
    stats: dict[str, dict] = {}
    for t in memory.trajectories():
        for eid in t.edge_ids:
            edge = graph.get_edge(eid)
            st = stats.setdefault(edge.mechanism_tag, {"n": 0, "improved": 0})
            st["n"] += 1
            if edge.delta is not None and (edge.delta > 0) == maximize:
                st["improved"] += 1
    added = 0
    for tag, st in stats.items():
        gen_score = st["improved"] / st["n"]
        if st["n"] >= min_support and st["improved"] > 0:
            pattern_memory.upsert_mechanism(
                mechanism_tag=tag,
                text=f"Mechanism '{tag}' improved fitness in {st['improved']}/{st['n']} observed uses.",
                generalization_score=gen_score,
                support_id=-1,
                updated_iter=iteration,
            )
            added += 1
        # anti-pattern 早停：尝试够多却几乎从不改进的机制，标记降权（接回 selection/operator）
        if st["n"] >= max(min_support, 5) and gen_score < 0.2 and not pattern_memory.is_anti_pattern(tag):
            pattern_memory.add(
                kind="anti_pattern",
                text=f"Mechanism '{tag}' improved only {gen_score:.0%} of {st['n']} uses; deprioritize.",
                mechanism_tag=tag,
                support_ids=(-1,),
                generalization_score=gen_score,
                confidence=0.8,
                updated_iter=iteration,
            )
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
            text=(f"Mechanism '{best['mechanism_tag']}' consistently yields strong fitness "
                  f"(idea: {best['idea'][:80]}). Prefer variants of it."),
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

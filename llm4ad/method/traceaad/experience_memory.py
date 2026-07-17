"""ExperienceMemory —— DerivationGraph 上的只读边级经验视图。

检索当前 run 内典型成功/失败 action，不复制图边，不维护第二份事实。
"""
from __future__ import annotations

import re

from .derivation_graph import DerivationGraph
from .schema import ExperienceBatch, ExperienceExample, ImprovementEdge

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_action_text(action: str) -> str:
    return _WHITESPACE_RE.sub(" ", action.strip())


class ExperienceMemory:
    def __init__(self, graph: DerivationGraph) -> None:
        self._graph = graph

    def examples(
        self,
        *,
        operator: str,
        positive_k: int = 2,
        negative_k: int = 2,
    ) -> ExperienceBatch:
        edges = [
            edge
            for edge in self._graph.edges()
            if edge.action and normalize_action_text(edge.action)
            and edge.delta is not None
            and edge.outcome in {"improve", "regress"}
        ]
        edges = self._deduplicate(edges, operator=operator)
        positives = self._select(
            edges,
            outcome="improve",
            operator=operator,
            k=max(0, int(positive_k)),
            reverse_delta=True,
        )
        negatives = self._select(
            edges,
            outcome="regress",
            operator=operator,
            k=max(0, int(negative_k)),
            reverse_delta=False,
        )
        return ExperienceBatch(positives=positives, negatives=negatives)

    @staticmethod
    def _deduplicate(
        edges: list[ImprovementEdge],
        *,
        operator: str,
    ) -> list[ImprovementEdge]:
        """Keep one representative edge for each normalized action."""
        selected: dict[str, ImprovementEdge] = {}
        for edge in edges:
            key = normalize_action_text(edge.action)
            current = selected.get(key)
            if current is None or ExperienceMemory._representative_rank(
                edge, operator
            ) > ExperienceMemory._representative_rank(current, operator):
                selected[key] = edge
        return list(selected.values())

    @staticmethod
    def _representative_rank(edge: ImprovementEdge, operator: str) -> tuple:
        return (
            edge.operator == operator,
            abs(float(edge.delta)),
            edge.iteration if edge.iteration is not None else -1,
            edge.id,
        )

    def _select(
        self,
        edges: list[ImprovementEdge],
        *,
        outcome: str,
        operator: str,
        k: int,
        reverse_delta: bool,
    ) -> tuple[ExperienceExample, ...]:
        if k <= 0:
            return ()
        matched = [edge for edge in edges if edge.outcome == outcome]
        operator_first = [edge for edge in matched if edge.operator == operator]
        others = [edge for edge in matched if edge.operator != operator]
        ranked = self._rank(operator_first, reverse_delta=reverse_delta) + self._rank(
            others, reverse_delta=reverse_delta
        )
        selected: list[ExperienceExample] = []
        for edge in ranked:
            selected.append(
                ExperienceExample(
                    edge_id=edge.id,
                    operator=edge.operator,
                    action=normalize_action_text(edge.action),
                    delta=float(edge.delta),
                    outcome=edge.outcome,
                    iteration=edge.iteration,
                )
            )
            if len(selected) >= k:
                break
        return tuple(selected)

    @staticmethod
    def _rank(
        edges: list[ImprovementEdge],
        *,
        reverse_delta: bool,
    ) -> list[ImprovementEdge]:
        return sorted(
            edges,
            key=lambda edge: (
                float(edge.delta) if reverse_delta else -float(edge.delta),
                edge.iteration if edge.iteration is not None else -1,
                edge.id,
            ),
            reverse=True,
        )

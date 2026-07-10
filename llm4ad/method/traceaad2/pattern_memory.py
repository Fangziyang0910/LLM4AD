"""PatternMemory —— 知识层记忆（蒸馏出的机制/教训/反模式）。

由 reflection（蒸馏回路 + 反思回路）写入，供 context（因果叙事 §6.B）按 mechanism_tag 检索
注入 prompt。本模块只负责存储/检索/淘汰，蒸馏逻辑在 reflection.py。
"""
from __future__ import annotations

from .schema import Pattern, PatternId


class PatternMemory:
    def __init__(self, *, capacity_per_kind: int = 50, min_confidence: float = 0.2) -> None:
        self.capacity_per_kind = capacity_per_kind
        self.min_confidence = min_confidence
        self._next_id = 0
        self._patterns: dict[PatternId, Pattern] = {}
        self._mechanism_outcomes: dict[tuple[str, str, int], tuple[bool, int]] = {}
        self._operator_scopes: dict[PatternId, str | None] = {}

    def add(self, *, kind: str, text: str, mechanism_tag: str, support_ids: tuple[int, ...] = (),
             generalization_score: float = 0.0, confidence: float = 1.0,
             updated_iter: int = 0, operator: str | None = None) -> Pattern:
        scope = None if operator is None else str(operator)
        existing: Pattern | None = None
        if kind == "lesson":
            existing = next((
                p for p in self._patterns.values()
                if p.kind == kind
                and p.mechanism_tag == mechanism_tag
                and self._operator_scopes.get(p.id) == scope
            ), None)
        elif kind == "anti_pattern":
            existing = next((
                p for p in self._patterns.values()
                if p.kind == kind
                and p.mechanism_tag == mechanism_tag
                and self._operator_scopes.get(p.id) == scope
            ), None)
        if existing is not None:
            merged = Pattern(
                id=existing.id,
                kind=existing.kind,
                text=text.strip() or existing.text,
                mechanism_tag=existing.mechanism_tag,
                support_ids=tuple(dict.fromkeys((*existing.support_ids, *support_ids)))[:50],
                generalization_score=generalization_score,
                confidence=confidence,
                updated_iter=updated_iter,
            )
            self._patterns[existing.id] = merged
            return merged
        pattern = Pattern(
            id=self._next_id,
            kind=kind,
            text=text,
            mechanism_tag=mechanism_tag,
            support_ids=support_ids,
            generalization_score=generalization_score,
            confidence=confidence,
            updated_iter=updated_iter,
        )
        self._patterns[pattern.id] = pattern
        self._operator_scopes[pattern.id] = scope
        self._next_id += 1
        self._prune_kind(kind)
        return pattern

    def upsert_mechanism(self, *, mechanism_tag: str, text: str, generalization_score: float,
                          support_id: int, updated_iter: int) -> Pattern:
        """机制模式的增量更新：同类合并 support，刷新泛化分数。"""
        existing = self.mechanism_pattern(mechanism_tag)
        if existing is None:
            return self.add(
                kind="mechanism",
                text=text,
                mechanism_tag=mechanism_tag,
                support_ids=(support_id,),
                generalization_score=generalization_score,
                confidence=1.0,
                updated_iter=updated_iter,
            )
        merged = Pattern(
            id=existing.id,
            kind=existing.kind,
            text=text if text else existing.text,
            mechanism_tag=existing.mechanism_tag,
            support_ids=tuple(dict.fromkeys((*existing.support_ids, support_id))),
            generalization_score=generalization_score,
            confidence=min(1.0, existing.confidence + 0.1),
            updated_iter=updated_iter,
        )
        self._patterns[existing.id] = merged
        return merged

    def mechanism_pattern(self, mechanism_tag: str) -> Pattern | None:
        for p in self._patterns.values():
            if p.kind == "mechanism" and p.mechanism_tag == mechanism_tag:
                return p
        return None

    def top_lessons(
        self,
        *,
        mechanism_tag: str | None = None,
        operator: str | None = None,
        k: int = 3,
    ) -> tuple[Pattern, ...]:
        scope = None if operator is None else str(operator)
        candidates = [
            p for p in self._patterns.values()
            if p.kind == "lesson"
            or (
                p.kind == "anti_pattern"
                and (
                    self._operator_scopes.get(p.id) is None
                    or (scope is not None and self._operator_scopes.get(p.id) == scope)
                )
            )
        ]
        if mechanism_tag is not None:
            tagged = [p for p in candidates if p.mechanism_tag == mechanism_tag]
            others = [p for p in candidates if p.mechanism_tag != mechanism_tag]
            ordered = tagged + others
        else:
            ordered = candidates
        ordered.sort(key=lambda p: (p.confidence * max(p.generalization_score, 0.1)), reverse=True)
        return tuple(ordered[:k])

    def top_mechanisms(self, *, k: int = 5) -> tuple[Pattern, ...]:
        mechs = [p for p in self._patterns.values() if p.kind == "mechanism"]
        mechs.sort(key=lambda p: (p.generalization_score, len(p.support_ids)), reverse=True)
        return tuple(mechs[:k])

    def record_mechanism_outcome(
        self,
        *,
        operator: str,
        mechanism_tag: str,
        support_id: int,
        success: bool,
        iteration: int,
    ) -> None:
        """Record idempotent operator-conditioned evidence for one real graph object."""
        key = (str(operator), mechanism_tag, support_id)
        self._mechanism_outcomes[key] = (success, iteration)

    def mechanism_attempts(self, mechanism_tag: str, *, operator: str | None = None) -> int:
        op = None if operator is None else str(operator)
        return sum(
            1
            for evidence_op, evidence_tag, _ in self._mechanism_outcomes
            if evidence_tag == mechanism_tag and (op is None or evidence_op == op)
        )

    def mechanism_successes(self, mechanism_tag: str, *, operator: str | None = None) -> int:
        op = None if operator is None else str(operator)
        return sum(
            int(success)
            for (evidence_op, evidence_tag, _), (success, _) in self._mechanism_outcomes.items()
            if evidence_tag == mechanism_tag and (op is None or evidence_op == op)
        )

    def mechanism_improve_rate(
        self, mechanism_tag: str, *, operator: str | None = None
    ) -> float | None:
        """Operator-conditioned improve rate. None means no evidence was observed."""
        attempts = self.mechanism_attempts(mechanism_tag, operator=operator)
        if attempts:
            return self.mechanism_successes(mechanism_tag, operator=operator) / attempts
        if operator is not None:
            return None
        p = self.mechanism_pattern(mechanism_tag)
        return p.generalization_score if p is not None else None

    def mechanism_last_attempt_iteration(
        self, mechanism_tag: str, *, operator: str
    ) -> int | None:
        outcomes = self._conditioned_outcomes(mechanism_tag, operator=operator)
        return outcomes[-1][0] if outcomes else None

    def mechanism_failure_streak(self, mechanism_tag: str, *, operator: str) -> int:
        streak = 0
        for _, _, success in reversed(self._conditioned_outcomes(mechanism_tag, operator=operator)):
            if success:
                break
            streak += 1
        return streak

    def mechanism_in_failure_cooldown(
        self,
        mechanism_tag: str,
        *,
        operator: str,
        iteration: int,
        failure_limit: int = 2,
        cooldown: int = 24,
    ) -> bool:
        if failure_limit <= 0 or cooldown <= 0:
            return False
        if self.mechanism_failure_streak(mechanism_tag, operator=operator) < failure_limit:
            return False
        last_iteration = self.mechanism_last_attempt_iteration(
            mechanism_tag, operator=operator
        )
        return (
            last_iteration is not None
            and max(0, iteration - last_iteration) < cooldown
        )

    def is_anti_pattern(self, mechanism_tag: str, *, operator: str | None = None) -> bool:
        scope = None if operator is None else str(operator)
        return any(
            p.kind == "anti_pattern" and p.mechanism_tag == mechanism_tag
            and (
                self._operator_scopes.get(p.id) is None
                or (scope is not None and self._operator_scopes.get(p.id) == scope)
            )
            for p in self._patterns.values()
        )

    def clear_anti_pattern(self, mechanism_tag: str, *, operator: str | None = None) -> bool:
        scope = None if operator is None else str(operator)
        ids = [
            p.id for p in self._patterns.values()
            if p.kind == "anti_pattern"
            and p.mechanism_tag == mechanism_tag
            and self._operator_scopes.get(p.id) == scope
        ]
        for pattern_id in ids:
            self._patterns.pop(pattern_id, None)
            self._operator_scopes.pop(pattern_id, None)
        return bool(ids)

    def patterns(self) -> tuple[Pattern, ...]:
        return tuple(self._patterns.values())

    def _prune_kind(self, kind: str) -> None:
        items = [p for p in self._patterns.values() if p.kind == kind]
        if len(items) <= self.capacity_per_kind:
            return
        items.sort(key=lambda p: (p.confidence * max(p.generalization_score, 0.1)), reverse=True)
        keep = {p.id for p in items[: self.capacity_per_kind]}
        for p in items:
            if p.id not in keep:
                self._patterns.pop(p.id, None)
                self._operator_scopes.pop(p.id, None)

    def _conditioned_outcomes(
        self, mechanism_tag: str, *, operator: str
    ) -> list[tuple[int, int, bool]]:
        op = str(operator)
        outcomes = [
            (iteration, support_id, success)
            for (evidence_op, evidence_tag, support_id), (success, iteration)
            in self._mechanism_outcomes.items()
            if evidence_op == op and evidence_tag == mechanism_tag
        ]
        outcomes.sort(key=lambda item: (item[0], item[1]))
        return outcomes

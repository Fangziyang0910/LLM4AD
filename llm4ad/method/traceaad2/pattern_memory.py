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

    def add(self, *, kind: str, text: str, mechanism_tag: str, support_ids: tuple[int, ...] = (),
             generalization_score: float = 0.0, confidence: float = 1.0, updated_iter: int = 0) -> Pattern:
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
            support_ids=tuple(dict.fromkeys((*existing.support_ids, support_id)))[:50],
            generalization_score=max(existing.generalization_score, generalization_score),
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

    def top_lessons(self, *, mechanism_tag: str | None = None, k: int = 3) -> tuple[Pattern, ...]:
        candidates = [p for p in self._patterns.values() if p.kind in ("lesson", "anti_pattern")]
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

    def mechanism_improve_rate(self, mechanism_tag: str) -> float | None:
        """该机制的跨轨迹 improve rate（来自 distill 的 generalization_score）。None=未见。"""
        p = self.mechanism_pattern(mechanism_tag)
        return p.generalization_score if p is not None else None

    def is_anti_pattern(self, mechanism_tag: str) -> bool:
        return any(
            p.kind == "anti_pattern" and p.mechanism_tag == mechanism_tag
            for p in self._patterns.values()
        )

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

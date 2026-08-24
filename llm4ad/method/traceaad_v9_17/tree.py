"""Single-parent algorithm tree used by TraceAAD V9.17."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .schema import Algorithm

VIRTUAL_ROOT_ID = 0


class Tree:
    def __init__(self, *, maximize: bool) -> None:
        self.maximize = maximize
        self._algorithms = {
            VIRTUAL_ROOT_ID: Algorithm(VIRTUAL_ROOT_ID, None, None, None)
        }

    def algorithms(self) -> tuple[Algorithm, ...]:
        return tuple(self._algorithms.values())

    def valid_algorithms(self) -> tuple[Algorithm, ...]:
        return tuple(item for item in self.algorithms() if item.id != VIRTUAL_ROOT_ID)

    def get_algorithm(self, algorithm_id: int) -> Algorithm:
        return self._algorithms[algorithm_id]

    def quality(self, algorithm: Algorithm) -> float:
        assert algorithm.fitness is not None
        return algorithm.fitness if self.maximize else -algorithm.fitness

    def add_algorithm(
        self,
        *,
        code: str,
        fitness: float,
        parent_id: int,
        hypothesis_id: int,
        idea: str | None,
        diff: str,
        added: int,
        removed: int,
        result: str,
        created_by: str | None,
    ) -> Algorithm:
        if parent_id not in self._algorithms:
            raise KeyError(f"unknown parent algorithm: {parent_id}")
        algorithm = Algorithm(
            id=max(self._algorithms) + 1,
            code=code,
            fitness=fitness,
            parent_id=parent_id,
            hypothesis_id=hypothesis_id,
            idea=idea,
            diff=diff,
            added=added,
            removed=removed,
            result=result,
            created_by=created_by,
        )
        self._algorithms[algorithm.id] = algorithm
        return algorithm

    def root_algorithms(self) -> tuple[Algorithm, ...]:
        return tuple(
            item for item in self.valid_algorithms() if item.parent_id == VIRTUAL_ROOT_ID
        )

    def hypothesis_algorithms(self, hypothesis_id: int) -> tuple[Algorithm, ...]:
        return tuple(
            item for item in self.valid_algorithms() if item.hypothesis_id == hypothesis_id
        )

    def ancestor_ids(self, algorithm_id: int) -> tuple[int, ...]:
        path: list[int] = []
        current: int | None = algorithm_id
        while current is not None:
            path.append(current)
            current = self.get_algorithm(current).parent_id
        return tuple(reversed(path))

    def best(self) -> Algorithm | None:
        algorithms = self.valid_algorithms()
        return max(algorithms, key=self.quality) if algorithms else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximize": self.maximize,
            "algorithms": [asdict(item) for item in self.algorithms()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Tree:
        tree = cls(maximize=bool(payload["maximize"]))
        tree._algorithms = {
            item["id"]: Algorithm(**item) for item in payload["algorithms"]
        }
        return tree


__all__ = ["Tree", "VIRTUAL_ROOT_ID"]

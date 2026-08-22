"""The algorithm tree used by TraceAAD V9.16."""

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
        self._best_quality: float | None = None

    @property
    def virtual_root_id(self) -> int:
        return VIRTUAL_ROOT_ID

    def algorithms(self) -> tuple[Algorithm, ...]:
        return tuple(self._algorithms.values())

    def valid_algorithms(self) -> tuple[Algorithm, ...]:
        return tuple(
            algorithm
            for algorithm in self._algorithms.values()
            if algorithm.id != VIRTUAL_ROOT_ID
        )

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
        parent_id: int = VIRTUAL_ROOT_ID,
        idea: str | None = None,
        created_by: str | None = None,
        entry_id: int | None = None,
    ) -> Algorithm:
        algorithm = Algorithm(
            id=max(self._algorithms) + 1,
            code=code,
            fitness=fitness,
            parent_id=parent_id,
            idea=idea,
            created_by=created_by,
            entry_id=entry_id,
        )
        self._algorithms[algorithm.id] = algorithm
        quality = self.quality(algorithm)
        if self._best_quality is None or quality > self._best_quality:
            self._best_quality = quality
        return algorithm

    def root_algorithms(self) -> tuple[Algorithm, ...]:
        return tuple(
            algorithm
            for algorithm in self.valid_algorithms()
            if algorithm.parent_id == VIRTUAL_ROOT_ID
        )

    def ancestor_ids(self, algorithm_id: int) -> tuple[int, ...]:
        path: list[int] = []
        current: int | None = algorithm_id
        while current is not None:
            path.append(current)
            current = self.get_algorithm(current).parent_id
        return tuple(reversed(path))

    def depth(self, algorithm_id: int) -> int:
        """Number of edges from the virtual root; roots have depth one."""
        return len(self.ancestor_ids(algorithm_id)) - 1

    def best(self) -> Algorithm | None:
        algorithms = self.valid_algorithms()
        return max(algorithms, key=self.quality) if algorithms else None

    def best_quality(self) -> float | None:
        return self._best_quality

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximize": self.maximize,
            "algorithms": [asdict(item) for item in self.algorithms()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Tree:
        tree = cls(maximize=payload["maximize"])
        tree._algorithms = {
            item["id"]: Algorithm(**item) for item in payload["algorithms"]
        }
        for algorithm in tree.valid_algorithms():
            quality = tree.quality(algorithm)
            if tree._best_quality is None or quality > tree._best_quality:
                tree._best_quality = quality
        return tree


__all__ = ["Tree", "VIRTUAL_ROOT_ID"]

"""The algorithm tree used by TraceAAD V9.19."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .schema import Algorithm

VIRTUAL_ROOT_ID = 0


class Tree:
    """Unique-parent algorithm tree; all five tasks maximize fitness."""

    def __init__(self) -> None:
        self._algorithms = {
            VIRTUAL_ROOT_ID: Algorithm(VIRTUAL_ROOT_ID, None, None, None)
        }
        self._best_quality: float | None = None

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
        return algorithm.fitness

    def add_algorithm(
        self,
        *,
        code: str,
        fitness: float,
        parent_id: int = VIRTUAL_ROOT_ID,
        idea: str | None = None,
        action: str | None = None,
        created_slot: int = 0,
        t_response: float = 0.5,
        novelty: float | None = None,
        behavior_tag: str | None = None,
    ) -> Algorithm:
        algorithm = Algorithm(
            id=max(self._algorithms) + 1,
            code=code,
            fitness=fitness,
            parent_id=parent_id,
            idea=idea,
            action=action,
            created_slot=created_slot,
            t_response=t_response,
            novelty=novelty,
            behavior_tag=behavior_tag,
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

    def formation_path(self, algorithm_id: int) -> tuple[int, ...]:
        """Nodes on the unique formation path, oldest first; roots are empty.

        Each real formation edge is represented by its child; the initial
        root has no formation edge.
        """
        path: list[int] = []
        current = algorithm_id
        while current != VIRTUAL_ROOT_ID:
            node = self.get_algorithm(current)
            if node.parent_id == VIRTUAL_ROOT_ID:
                break
            path.append(current)
            current = node.parent_id
        return tuple(reversed(path))

    def best(self) -> Algorithm | None:
        algorithms = self.valid_algorithms()
        return max(algorithms, key=self.quality) if algorithms else None

    def best_quality(self) -> float | None:
        return self._best_quality

    def to_dict(self) -> dict[str, Any]:
        return {"algorithms": [asdict(item) for item in self.algorithms()]}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Tree:
        tree = cls()
        tree._algorithms = {
            item["id"]: Algorithm(**item) for item in payload["algorithms"]
        }
        for algorithm in tree.valid_algorithms():
            quality = tree.quality(algorithm)
            if tree._best_quality is None or quality > tree._best_quality:
                tree._best_quality = quality
        return tree


__all__ = ["Tree", "VIRTUAL_ROOT_ID"]

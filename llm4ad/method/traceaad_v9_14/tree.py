"""Single Tree data structure holding Algorithm nodes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

from .schema import Algorithm, Outcome

VIRTUAL_ROOT_ID = 0


def is_better(candidate: Algorithm, incumbent: Algorithm | None) -> bool:
    """判断 candidate 是否严格优于 incumbent (同分时代码短者优先，次之创建更早者优先)。"""
    if incumbent is None:
        return True
    assert candidate.q is not None and incumbent.q is not None
    assert candidate.code is not None and incumbent.code is not None
    return (candidate.q, -len(candidate.code), -candidate.id) > (
        incumbent.q,
        -len(incumbent.code),
        -incumbent.id,
    )


class Tree:
    """管理单根算法演化树（只包含有效 Algorithm 节点）。"""

    def __init__(self, *, maximize: bool) -> None:
        self.maximize = maximize
        self.virtual_root_id = VIRTUAL_ROOT_ID
        self._algorithms: dict[int, Algorithm] = {}
        self.branch_ids: list[int] = []  # 初始 Level-1 算法的 ID 列表
        self._next_algorithm_id = 0

        # 初始化虚拟根节点 (Level-0)
        self._init_virtual_root()

    def _init_virtual_root(self) -> None:
        virtual_root = Algorithm(
            id=self.virtual_root_id,
            code=None,
            fitness=None,
            q=None,
            parent_id=None,
            count=0,
        )
        self._algorithms[self.virtual_root_id] = virtual_root
        self._next_algorithm_id = self.virtual_root_id + 1

    def algorithms(self) -> tuple[Algorithm, ...]:
        """返回所有算法节点（包含虚拟根节点）。"""
        return tuple(self._algorithms.values())

    def valid_algorithms(self) -> tuple[Algorithm, ...]:
        """返回所有真实有效算法节点（排除虚拟根节点）。"""
        return tuple(
            algo for algo in self._algorithms.values() if algo.id != self.virtual_root_id
        )

    def get_algorithm(self, algorithm_id: int) -> Algorithm:
        return self._algorithms[algorithm_id]

    def add_algorithm(
        self,
        *,
        code: str,
        fitness: float,
        parent_id: int | None = None,
        **kwargs: Any,
    ) -> Algorithm:
        """添加算法节点（自动分配 ID、计算标准化 q、维护树关系）。"""
        algo_id = self._next_algorithm_id
        self._next_algorithm_id += 1
        algo = Algorithm(
            id=algo_id,
            code=code,
            fitness=fitness,
            q=fitness if self.maximize else -fitness,
            parent_id=self.virtual_root_id if parent_id is None else parent_id,
            **kwargs,
        )
        self._algorithms[algo.id] = algo
        if parent_id is None:
            self.branch_ids.append(algo.id)
        return algo

    def add_branch_root(self, *, code: str, fitness: float, stage: str = "root_generation") -> Algorithm:
        return self.add_algorithm(code=code, fitness=fitness, parent_id=None, stage=stage)

    def add_child(self, *, parent_id: int, code: str, fitness: float, **kwargs: Any) -> Algorithm:
        return self.add_algorithm(code=code, fitness=fitness, parent_id=parent_id, **kwargs)


    def ancestor_ids(self, algorithm_id: int) -> tuple[int, ...]:
        """获取从虚拟根节点到当前算法节点的完整祖先路径 ID。"""
        path: list[int] = []
        current: int | None = algorithm_id
        while current is not None:
            path.append(current)
            current = self.get_algorithm(current).parent_id
        return tuple(reversed(path))

    def branch_id_of(self, algorithm_id: int) -> int:
        """查询该算法节点属于哪个 Level-1 主分支。"""
        ancestors = self.ancestor_ids(algorithm_id)
        if len(ancestors) < 2:
            return self.virtual_root_id
        return ancestors[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximize": self.maximize,
            "virtual_root_id": self.virtual_root_id,
            "next_algorithm_id": self._next_algorithm_id,
            "branch_ids": list(self.branch_ids),
            "algorithms": [asdict(item) for item in self.algorithms()],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Tree:
        tree = cls(maximize=bool(payload["maximize"]))
        tree.virtual_root_id = int(payload.get("virtual_root_id", VIRTUAL_ROOT_ID))
        tree._algorithms.clear()

        for item in payload["algorithms"]:
            if item.get("outcome") is not None:
                item = {**item, "outcome": Outcome(item["outcome"])}
            algorithm = Algorithm(**item)
            tree._algorithms[algorithm.id] = algorithm
        tree.branch_ids = list(payload.get("branch_ids", []))
        tree._next_algorithm_id = int(payload["next_algorithm_id"])
        return tree


__all__ = ["Tree", "VIRTUAL_ROOT_ID", "is_better"]

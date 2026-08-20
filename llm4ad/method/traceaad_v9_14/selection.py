"""Two-level UCB-style allocation on the Algorithm Tree."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .tree import Tree


@dataclass(frozen=True, slots=True)
class BranchScore:
    """Level-1 主分支打分。"""
    branch_id: int   # 分支根算法 ID
    q: float         # 该分支内达到的历史最佳质量
    count: int       # 该分支累计消耗的变异次数
    optimism: float  # UCB 乐观探索项
    score: float     # 综合总分


@dataclass(frozen=True, slots=True)
class AlgorithmScore:
    """分支内部具体算法节点打分。"""
    algorithm_id: int  # 算法 ID
    q: float           # 算法自身质量
    count: int         # 算法被选中的累计次数
    optimism: float    # UCB 乐观探索项
    score: float       # 综合总分


@dataclass(frozen=True, slots=True)
class SelectionChoice:
    """分配器做出的最终选择。"""
    algorithm_id: int                       # 选中的目标算法 ID
    branch_id: int                          # 选中的主分支 ID
    branches: tuple[BranchScore, ...]       # 所有分支的打分快照
    algorithms: tuple[AlgorithmScore, ...]  # 选中分支内所有算法的打分快照


def score_branches(tree: Tree, s: float) -> tuple[BranchScore, ...]:
    """计算每个 Level-1 主分支的 UCB 得分: q*(branch) + s / sqrt(N(branch) + 1)。"""
    best_q: dict[int, float] = {}
    spent_count: dict[int, int] = {}

    for algo in tree.valid_algorithms():
        assert algo.q is not None
        b_id = tree.branch_id_of(algo.id)
        best_q[b_id] = max(best_q.get(b_id, -float("inf")), algo.q)
        spent_count[b_id] = spent_count.get(b_id, 0) + algo.count

    result: list[BranchScore] = []
    for b_id in tree.branch_ids:
        q = best_q[b_id]
        count = spent_count.get(b_id, 0)
        opt = s / math.sqrt(count + 1)
        result.append(BranchScore(b_id, q, count, opt, q + opt))
    return tuple(result)


def score_algorithms_in_branch(
    tree: Tree, s: float, selected_branch: int
) -> tuple[AlgorithmScore, ...]:
    """在选中的主分支内计算每个算法节点的 UCB 得分: q(algo) + s / sqrt(count(algo) + 1)。"""
    scored: list[AlgorithmScore] = []
    for algo in tree.valid_algorithms():
        if tree.branch_id_of(algo.id) == selected_branch:
            assert algo.q is not None
            opt = s / math.sqrt(algo.count + 1)
            scored.append(AlgorithmScore(algo.id, algo.q, algo.count, opt, algo.q + opt))
    return tuple(scored)


def select(tree: Tree, s: float) -> SelectionChoice:
    """两级选择：先选分支，再在分支内选具体待改进的算法。"""
    branches = score_branches(tree, s)
    if not branches:
        raise ValueError("cannot allocate budget without an algorithm")

    # 分支/算法排序：分数最高优先；平局时访问更少优先，次之创建更早优先
    chosen_branch = max(branches, key=lambda b: (b.score, -b.count, -b.branch_id))
    algorithms = score_algorithms_in_branch(tree, s, chosen_branch.branch_id)
    chosen_algo = max(algorithms, key=lambda a: (a.score, -a.count, -a.algorithm_id))

    return SelectionChoice(
        algorithm_id=chosen_algo.algorithm_id,
        branch_id=chosen_branch.branch_id,
        branches=branches,
        algorithms=algorithms,
    )


__all__ = [
    "AlgorithmScore",
    "BranchScore",
    "SelectionChoice",
    "score_algorithms_in_branch",
    "score_branches",
    "select",
]

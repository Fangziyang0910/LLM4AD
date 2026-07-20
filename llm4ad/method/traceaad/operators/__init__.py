"""算子集。"""
from __future__ import annotations

from .base import (
    Operator,
    OperatorContext,
    branch_score,
    classify_outcome,
    select_base_node,
    trajectory_step_outcomes,
)
from .backtrack import BacktrackBranchOp
from .crossover import MechanismCrossoverOp
from .endpoint import EndpointRefineOp
from .novelty import NoveltyJumpOp

DEFAULT_OPERATORS: tuple[type[Operator], ...] = (
    EndpointRefineOp,
    BacktrackBranchOp,
    MechanismCrossoverOp,
    NoveltyJumpOp,
)

__all__ = [
    "Operator",
    "OperatorContext",
    "DEFAULT_OPERATORS",
    "EndpointRefineOp",
    "BacktrackBranchOp",
    "MechanismCrossoverOp",
    "NoveltyJumpOp",
    "select_base_node",
    "branch_score",
    "classify_outcome",
    "trajectory_step_outcomes",
]

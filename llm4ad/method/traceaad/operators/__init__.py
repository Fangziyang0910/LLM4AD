"""算子集（design §4/§7）。"""
from __future__ import annotations

from .base import (
    Operator,
    OperatorContext,
    branch_score,
    classify_outcome,
    infer_mechanism_tag,
    select_base_node,
)
from .backtrack import BacktrackBranchOp
from .crossover import MechanismCrossoverOp
from .endpoint import EndpointRefineOp
from .novelty import NoveltyJumpOp
from .scale_transfer import ScaleTransferOp
from .simplify import DistillSimplifyOp

DEFAULT_OPERATORS: tuple[type[Operator], ...] = (
    EndpointRefineOp,
    BacktrackBranchOp,
    MechanismCrossoverOp,
    DistillSimplifyOp,
    ScaleTransferOp,
    NoveltyJumpOp,
)

__all__ = [
    "Operator",
    "OperatorContext",
    "DEFAULT_OPERATORS",
    "EndpointRefineOp",
    "BacktrackBranchOp",
    "MechanismCrossoverOp",
    "DistillSimplifyOp",
    "ScaleTransferOp",
    "NoveltyJumpOp",
    "select_base_node",
    "branch_score",
    "infer_mechanism_tag",
    "classify_outcome",
]

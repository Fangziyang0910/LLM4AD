"""Scale-Transfer —— generalize（design §4.5，新增算子，直驱泛化目标）。

让当前机制更 scale-invariant / instance-agnostic，使其跨规模与分布可迁移。
只有 evaluation 明确提供 held-out、per-instance 或跨规模证据时才允许触发；默认
robustness 不构成泛化证据。
"""
from __future__ import annotations

from ..schema import OperatorName
from .base import OperatorContext, _ExtendFromEndpointOp


class ScaleTransferOp(_ExtendFromEndpointOp):
    name = OperatorName.SCALE_TRANSFER
    role = "generalize"

    def trigger(self, ctx: OperatorContext) -> bool:
        node = ctx.graph.get_node(ctx.selected.endpoint_id)
        if not node.is_valid or node.fitness is None:
            return False
        return ctx.has_generalization_evidence

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        ctx.hints["mechanism_tag_hint"] = "generalize"
        return (
            "Generalize: make the current mechanism more scale-invariant and instance-agnostic so it "
            "transfers across problem sizes and instance distributions. Prefer closed-form, "
            "parameter-light formulations over instance-specific hardcoding or magic constants."
        )

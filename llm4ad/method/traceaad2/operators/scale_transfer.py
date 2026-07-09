"""Scale-Transfer —— generalize（design §4.5，新增算子，直驱泛化目标）。

让当前机制更 scale-invariant / instance-agnostic，使其跨规模与分布可迁移。
当前平台 task 只返回标量，无法真换规模评估；这里通过 prompt 约束 + 高 robustness 触发
+ 泛化信用加权，间接驱动泛化（未来 task 支持 per-instance/多规模时可直接评估迁移效果）。
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
        # 高 robustness（疑似可泛化）或周期性触发，鼓励把好机制推向更通用
        return node.robustness >= 0.5 or (ctx.iteration % 7 == 0 and ctx.iteration > 0)

    def build_constraint(self, ctx: OperatorContext, base_node_id: int | None) -> str:
        ctx.hints["mechanism_tag_hint"] = "generalize"
        return (
            "Generalize: make the current mechanism more scale-invariant and instance-agnostic so it "
            "transfers across problem sizes and instance distributions. Prefer closed-form, "
            "parameter-light formulations over instance-specific hardcoding or magic constants."
        )

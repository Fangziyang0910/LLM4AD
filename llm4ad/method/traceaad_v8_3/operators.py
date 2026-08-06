"""The five V8.3 trajectory operators."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import OperatorName


@dataclass(frozen=True, slots=True)
class Operator:
    name: OperatorName
    instruction: str


DEFAULT_OPERATORS = (
    Operator(
        OperatorName.REFINE,
        "聚焦于继续发展或修复当前机制。根据来时路中已经形成的有效思路、当前算法的薄弱点以及已尝试分支暴露的问题，选择一个最值得继续开发的方向，提出聚焦且完整的下一步改进。",
    ),
    Operator(
        OperatorName.TUNE,
        "聚焦于校准当前机制的参数、阈值、尺度、触发条件或控制细节。判断哪些细节限制了当前算法，并为校准它们做出必要的配套修改；配套修改可以包含状态统计、归一化、自适应控制或局部结构调整。",
    ),
    Operator(
        OperatorName.SIMPLIFY,
        "聚焦于降低当前算法不必要的机制和实现复杂度。判断哪些部分可以删除、合并、重组，或用更简单的机制取代，同时保留当前算法的关键功能。简化应当减少真实概念或代码复杂度，不是只改名、压缩排版或隐藏相同逻辑。",
    ),
    Operator(
        OperatorName.INNOVATE,
        "聚焦于从当前节点探索一个明显不同的核心思路。利用局部探索脉络识别已经停滞或反复失败的方向，提出具有实质机制差异的新路线。可以保留当前算法中仍然有价值的组件，但不应将普通的局部调整当作换新方向。",
    ),
    Operator(
        OperatorName.CROSSOVER,
        "以当前算法为主体，从参考算法中识别一项与当前机制互补的思想，将它选择性地适配并融入当前代码。判断两者的功能关系和冲突，完成必要的配套调整。不要机械拼接两份完整代码，也不要无选择地复制整个参考算法。",
    ),
)


__all__ = ["DEFAULT_OPERATORS", "Operator"]

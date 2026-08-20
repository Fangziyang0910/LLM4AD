"""Protocol constants and data structures for TraceAAD V9.14."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

PROTOCOL_ID: Final[str] = "traceaad-v9.14-tree-algorithm"
REFINE_PROBABILITY: Final[float] = 0.7
INITIAL_ROOT_COUNT: Final[int] = 8
MAX_HISTORY_EVENTS: Final[int] = 8
LOGICAL_MODEL_NAME: Final[str] = "Qwen3.6-27B"


class Intent(StrEnum):
    """LLM 生成意图。"""
    REFINE = "refine"    # 聚焦于现有机制的局部改进
    EXPLORE = "explore"  # 尝试不同设计或大幅重构


class Outcome(StrEnum):
    """单步演化的定性结果。"""
    IMPROVE = "improve"  # 性能提升 (dq > 0)
    PLATEAU = "plateau"  # 性能持平 (dq == 0)
    REGRESS = "regress"  # 性能下降 (dq < 0)


@dataclass(slots=True)
class Algorithm:
    """搜索树上的算法节点（承载算法代码、质量及自身生成事实）。"""
    id: int                        # 算法唯一整数 ID (自增，天然具备创建次序)
    code: str | None               # 算法完整 Python 代码 (虚拟根节点为 None)
    fitness: float | None          # 真实适应度数值 (虚拟根节点为 None)
    q: float | None                # 标准化质量分 (统一转化为越大越好，虚拟根节点为 None)
    parent_id: int | None          # 父算法节点 ID (虚拟根节点为 None)
    count: int = 0                 # 该算法被选为父代发起变异的累计次数
    # 以下为生成该算法节点时的变异元数据 (虚拟根与初始根算法为 None/默认值)
    intent: str | None = None      # 生成该节点时的意图 (refine / explore)
    idea: str | None = None        # 模型声明的设计思想
    diff: str | None = None        # 与父算法代码的差异文本 (diff)
    added: int = 0                 # 相对父算法增加的代码行数
    removed: int = 0               # 相对父算法删除的代码行数
    dq: float | None = None        # 相对父算法的质量变化量 (q_self - q_parent)
    outcome: Outcome | None = None # 定性演化结果 (improve / plateau / regress)
    stage: str | None = None       # 诞生阶段: 'root_generation', 'bootstrap', 'search'
    iteration: int | None = None   # 诞生时的正式搜索迭代轮次


@dataclass(slots=True)
class Pending:
    """暂存已生成但尚未完成评价落盘的候选响应。"""
    order: int
    parent_id: int | None
    stage: str
    iteration: int | None
    intent: str | None
    response: str


__all__ = [
    "INITIAL_ROOT_COUNT",
    "LOGICAL_MODEL_NAME",
    "MAX_HISTORY_EVENTS",
    "PROTOCOL_ID",
    "REFINE_PROBABILITY",
    "Algorithm",
    "Intent",
    "Outcome",
    "Pending",
]

"""Parent-child code diff computation."""

from __future__ import annotations

import difflib


def code_diff(parent: str, child: str) -> tuple[str, int, int]:
    """计算父子算法之间的代码差异，并返回 (diff文本, 增加行数, 删除行数)。"""
    lines = list(
        difflib.unified_diff(
            parent.splitlines(),
            child.splitlines(),
            fromfile="parent.py",
            tofile="candidate.py",
            lineterm="",
        )
    )
    added = sum(line.startswith("+") and not line.startswith("+++") for line in lines)
    removed = sum(line.startswith("-") and not line.startswith("---") for line in lines)
    return "\n".join(lines), added, removed


__all__ = ["code_diff"]

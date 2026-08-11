"""Exact source identity and deterministic parent-child diffs."""

from __future__ import annotations

import difflib
import hashlib

from .schema import DiffStatistics


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def nonempty_loc(code: str) -> int:
    return sum(bool(line.strip()) for line in code.splitlines())


def actual_code_diff(parent: str, child: str) -> tuple[str, DiffStatistics]:
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
    return "\n".join(lines), DiffStatistics(
        added_lines=added,
        removed_lines=removed,
        changed_lines=added + removed,
    )


__all__ = ["actual_code_diff", "nonempty_loc", "text_hash"]

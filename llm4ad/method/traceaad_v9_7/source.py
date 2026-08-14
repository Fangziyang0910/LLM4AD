"""Code identity and parent-child diffs."""

from __future__ import annotations

import difflib
import hashlib


def code_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def code_diff(parent: str, child: str) -> tuple[str, int, int]:
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


__all__ = ["code_diff", "code_hash"]

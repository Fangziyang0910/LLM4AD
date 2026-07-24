"""Deterministic program normalization and lightweight complexity."""

from __future__ import annotations

import hashlib
import re


def normalized_source(code: str) -> str:
    text = code.replace("\r\n", "\n").replace("\r", "\n").strip()
    fenced = re.fullmatch(
        r"```(?:python|py)?\s*(?P<code>.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced is not None:
        text = fenced.group("code").strip()
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def nonempty_loc(code: str) -> int:
    return sum(bool(line.strip()) for line in normalized_source(code).splitlines())


def code_hash(code: str) -> str:
    return hashlib.sha256(normalized_source(code).encode("utf-8")).hexdigest()


def code_change_ratio(parent: str, child: str) -> float:
    left = normalized_source(parent).splitlines()
    right = normalized_source(child).splitlines()
    denominator = max(len(left), len(right), 1)
    shared = sum(a == b for a, b in zip(left, right))
    return 1.0 - shared / denominator


__all__ = ["code_change_ratio", "code_hash", "nonempty_loc", "normalized_source"]

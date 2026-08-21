"""Fitness formatting and one-line text compaction.

The CO ablation arm omits the parent-path history from prompts, so it needs
only these helpers; the diff rendering lives in the V9.7 arm."""

from __future__ import annotations


def format_fitness(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6g}"


def one_line(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


__all__ = ["format_fitness", "one_line"]

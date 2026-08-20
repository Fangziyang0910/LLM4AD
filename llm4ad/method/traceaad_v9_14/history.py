"""Parent improvement path extraction and rendering directly from Algorithm nodes."""

from __future__ import annotations

from .schema import MAX_HISTORY_EVENTS, Algorithm
from .tree import Tree

CHANGE_EXAMPLES_PER_SIDE = 2
CHANGE_LINE_MAX_CHARS = 520


def parent_path(
    tree: Tree, algorithm_id: int, *, max_events: int = MAX_HISTORY_EVENTS
) -> tuple[int, ...]:
    """提取该算法节点的父代改进形成路径（至多保留最近 max_events 条演化节点 ID）。"""
    ancestors = tree.ancestor_ids(algorithm_id)
    # ancestors[0] 是虚拟根, ancestors[1] 是初始根算法, ancestors[2:] 为演化生成的节点
    return ancestors[2:][-max_events:]


def drop_oldest(algorithm_ids: tuple[int, ...]) -> tuple[int, ...]:
    """超出上下文长度时，逐条舍弃最旧的形成事件。"""
    return algorithm_ids[1:]


def render_path(tree: Tree, algorithm_ids: tuple[int, ...]) -> str:
    """将父代形成节点序列渲染为 LLM 可读的文本块。"""
    lines = ["[Recent Algorithm Improvement History]"]
    if not algorithm_ids:
        lines.append("No history events are shown for this algorithm.")
        return "\n".join(lines)
    for index, algo_id in enumerate(algorithm_ids, start=1):
        algo = tree.get_algorithm(algo_id)
        parent = (
            None
            if algo.parent_id is None
            else tree.get_algorithm(algo.parent_id)
        )
        parent_fitness = None if parent is None else parent.fitness
        lines.append("")
        lines.append(f"[History {index}] Formation step")
        lines.extend(_render_event(algo, parent_fitness))
    return "\n".join(lines)


def _render_event(algo: Algorithm, parent_fitness: float | None) -> list[str]:
    assert algo.outcome is not None
    return [
        f"Idea: {algo.idea or 'unavailable'}",
        f"Change: {_compact_change(algo.diff, algo.added, algo.removed)}",
        f"Result: {algo.outcome.value}",
        (
            "Fitness: "
            f"{format_fitness(parent_fitness)} -> "
            f"{format_fitness(algo.fitness)}"
        ),
    ]


def _compact_change(diff: str | None, added: int, removed: int) -> str:
    if not diff:
        return "no recorded code change"
    removed_lines = _example_lines(_changed_code_lines(diff, "-"))
    added_lines = _example_lines(_changed_code_lines(diff, "+"))
    summary = (
        f"+{added}/-{removed} lines; "
        f"removed: {' | '.join(f'`{line}`' for line in removed_lines) or 'none'}; "
        f"added: {' | '.join(f'`{line}`' for line in added_lines) or 'none'}"
    )
    return one_line(summary, CHANGE_LINE_MAX_CHARS)


def _changed_code_lines(diff: str, prefix: str) -> list[str]:
    selected: list[str] = []
    seen: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith(prefix) or line.startswith(prefix * 3):
            continue
        code = line[1:].strip()
        if not code or code.startswith(("#", '"""', "'''")):
            continue
        normalized = one_line(code, 150)
        if normalized not in seen:
            selected.append(normalized)
            seen.add(normalized)
    return selected


def _example_lines(lines: list[str]) -> list[str]:
    if len(lines) <= CHANGE_EXAMPLES_PER_SIDE:
        return lines
    return [lines[0], lines[-1]]


def format_fitness(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:.6g}"


def one_line(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


__all__ = [
    "CHANGE_EXAMPLES_PER_SIDE",
    "CHANGE_LINE_MAX_CHARS",
    "drop_oldest",
    "format_fitness",
    "one_line",
    "parent_path",
    "render_path",
]

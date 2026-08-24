"""Trajectory-conditioned Refine, Explore, root, and repair prompts for V9.17."""

from __future__ import annotations

import ast
import copy
import re
from dataclasses import dataclass

from ...base import Function
from .history import format_fitness
from .schema import Intent

IDEA_TARGET_MAX_CHARS = 500

INTENT_INSTRUCTIONS: dict[Intent, str] = {
    Intent.REFINE: (
        "Diagnose the current algorithm's main limitation and continue developing "
        "its existing design. Preserve structures that the trajectory shows to be "
        "useful and make one coherent next improvement."
    ),
    Intent.EXPLORE: (
        "Propose a materially different algorithmic direction with a different "
        "core mechanism or decision logic. Do not merely tune parameters or make "
        "a small local modification."
    ),
}


@dataclass(frozen=True, slots=True)
class ParsedCandidate:
    declared_idea: str | None
    code: str


def fitness_direction_hint(maximize: bool) -> str:
    return (
        "Fitness is this task's score and higher is better."
        if maximize
        else "Fitness is this task's metric and lower is better."
    )


def build_root_prompt(
    *,
    task_description: str,
    template_function: Function,
    maximize: bool,
    error_handling: bool = False,
) -> str:
    target = copy.deepcopy(template_function)
    target.body = ""
    return "\n".join(
        [
            "[Task]",
            task_description.strip(),
            fitness_direction_hint(maximize),
            "",
            "[Target Function]",
            str(target).rstrip(),
            "Keep the function name, arguments, return type, and contract unchanged.",
            "Include every required import and helper in the returned program.",
            "",
            "[Instruction]",
            "Create one complete, valid, and competitive initial algorithm.",
            "Avoid a placeholder or trivial baseline.",
            *_reliability_contract(error_handling),
            _output_contract(),
        ]
    )


def build_generation_prompt(
    *,
    task_description: str,
    code: str,
    fitness: float,
    history_text: str,
    intent: Intent,
    maximize: bool,
    error_handling: bool = False,
) -> str:
    return "\n".join(
        [
            "[Task]",
            task_description.strip(),
            fitness_direction_hint(maximize),
            "",
            "[Current Algorithm]",
            f"Fitness: {format_fitness(fitness)}",
            "```python",
            code.rstrip(),
            "```",
            "",
            history_text.strip(),
            "",
            "[Instruction]",
            INTENT_INSTRUCTIONS[intent],
            "Keep the target function signature and contract unchanged.",
            "Return one complete, self-contained implementation.",
            *_reliability_contract(error_handling),
            _output_contract(),
        ]
    )


def build_repair_prompt(
    *,
    task_description: str,
    parent_code: str | None,
    failed_code: str,
    error: str,
    intent: Intent | None,
    maximize: bool,
    reliability: bool = True,
) -> str:
    intent_text = "initialization" if intent is None else intent.value
    parent_block = (
        "No evaluated parent code is available; repair the failed initial candidate."
        if parent_code is None
        else "[Evaluated Parent]\n```python\n" + parent_code.rstrip() + "\n```"
    )
    return "\n".join(
        [
            "[Task]",
            task_description.strip(),
            fitness_direction_hint(maximize),
            "",
            parent_block,
            "",
            "[Failed Candidate]",
            "```python",
            failed_code.rstrip(),
            "```",
            f"Failure during {intent_text}: {error}",
            "",
            "[Instruction]",
            "Repair the failed candidate with the smallest change that addresses the reported failure.",
            "Preserve the intended algorithmic idea and the target function signature.",
            *_reliability_contract(reliability),
            _output_contract(),
        ]
    )


def _output_contract() -> str:
    return "\n".join(
        [
            "Output one concise Idea and one complete Python program:",
            f"Idea: <one sentence, within {IDEA_TARGET_MAX_CHARS} characters>",
            "Code:",
            "```python",
            "<complete executable implementation>",
            "```",
        ]
    )


def _reliability_contract(enabled: bool) -> tuple[str, ...]:
    if not enabled:
        return ()
    return (
        "Reliability constraints:",
        "Keep execution bounded; do not use unbounded loops or recursion.",
        "Return a value that satisfies the target function contract on every call.",
        "Do not mutate input objects unless the task explicitly permits it.",
    )


def preflight_code(code: str, function_name: str) -> str | None:
    try:
        tree = ast.parse(code)
        compile(tree, "<traceaad-candidate>", "exec")
    except (SyntaxError, ValueError, TypeError) as exc:
        return f"preflight_syntax: {exc}"
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(definitions) != 1:
        return f"preflight_signature: expected one top-level function {function_name!r}"
    return None


_TIMEOUT_GUIDANCE = (
    "The failure is the per-call time limit: reduce the computational cost of "
    "each call (vectorize, precompute tables, shrink candidate lists) and keep "
    "the algorithmic idea."
)
_INTERNAL_FRAME_MARKERS = (
    "site-packages",
    "llm4ad",
    "lib/python",
    "multiprocessing",
    "concurrent/futures",
)
_MAX_FEEDBACK_FRAMES = 8


def _traceback_frame_blocks(traceback_text: str | None) -> list[str]:
    if not traceback_text:
        return []
    blocks: list[str] = []
    current: list[str] = []
    for line in traceback_text.splitlines():
        if line.lstrip().startswith('File "'):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def _candidate_frames(traceback_text: str | None) -> list[str]:
    blocks = _traceback_frame_blocks(traceback_text)
    kept = [
        block
        for block in blocks
        if not any(marker in block for marker in _INTERNAL_FRAME_MARKERS)
    ]
    if not kept:
        kept = blocks[-2:]
    return kept[-_MAX_FEEDBACK_FRAMES:]


def format_failure_feedback(
    *,
    error_type: str | None,
    error: str,
    traceback: str | None = None,
    preflight_error: str | None = None,
) -> str:
    head = error if not error_type else f"{error_type}: {error}"
    if preflight_error is not None:
        head = f"{head}\nPreflight: {preflight_error}"
    parts = [head]
    if error_type == "TimeoutError":
        parts.append(_TIMEOUT_GUIDANCE)
    frames = _candidate_frames(traceback)
    if frames:
        parts.append(
            "Traceback (candidate frames, innermost last):\n" + "\n".join(frames)
        )
    return "\n".join(parts)


def parse_program_response(response: str) -> ParsedCandidate:
    text = str(response)
    first_fence = text.find("```")
    blocks = tuple(
        block.strip()
        for block in re.findall(
            r"```(?:python|py)?\s*(.*?)(?:```|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if block.strip()
    )
    if blocks:
        return ParsedCandidate(extract_idea(text[:first_fence]), blocks[-1])
    marker = re.search(r"^\s*Code\s*:\s*", text, re.IGNORECASE | re.MULTILINE)
    if marker is not None:
        return ParsedCandidate(
            extract_idea(text[: marker.start()]), text[marker.end() :].strip()
        )
    return ParsedCandidate(extract_idea(text), text.strip())


def extract_idea(response: str) -> str | None:
    match = re.search(
        r"^\s*Idea\s*:\s*(?P<idea>\S[^\r\n]*)$",
        str(response),
        flags=re.IGNORECASE | re.MULTILINE,
    )
    return None if match is None else " ".join(match.group("idea").split())


__all__ = [
    "IDEA_TARGET_MAX_CHARS",
    "INTENT_INSTRUCTIONS",
    "ParsedCandidate",
    "build_generation_prompt",
    "build_repair_prompt",
    "build_root_prompt",
    "extract_idea",
    "fitness_direction_hint",
    "format_failure_feedback",
    "parse_program_response",
    "preflight_code",
]

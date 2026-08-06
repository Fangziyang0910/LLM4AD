"""Direct program generation and parsing protocol for TraceAAD V8.2."""

from __future__ import annotations

import re

from ...base import Program

from ..traceaad_v5.prompt import (
    IDEA_MAX_CHARS,
    ParsedProgram,
    build_initial_prompt,
    fitness_direction_hint,
    format_fitness,
    parse_program_response as _parse_program_response,
)


def parse_program_response(
    response: str,
    template_program: Program,
    function_name: str,
) -> ParsedProgram | None:
    """Require an explicit Idea because it is V8's only semantic edge record."""
    text = str(response)
    first_fence = text.find("```")
    if first_fence < 0:
        return None
    if re.search(
        r"^\s*Idea\s*:\s*\S[^\r\n]*$",
        text[:first_fence],
        flags=re.IGNORECASE | re.MULTILINE,
    ) is None:
        return None
    return _parse_program_response(text, template_program, function_name)

__all__ = [
    "IDEA_MAX_CHARS",
    "ParsedProgram",
    "build_initial_prompt",
    "fitness_direction_hint",
    "format_fitness",
    "parse_program_response",
]

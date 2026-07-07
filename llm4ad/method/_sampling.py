from __future__ import annotations

import re
from typing import Any, Callable

from ..base import SampleTrimmer


def log_sampler_llm_call(profiler: Any, **payload) -> None:
    logger = getattr(profiler, "log_llm_call", None)
    if callable(logger):
        try:
            logger(**payload)
        except Exception:
            pass


def trim_braced_thought(response: str) -> str | None:
    try:
        bracketed_texts = re.findall(r"\{.*?\}", response)
        return bracketed_texts[0]
    except Exception:
        return None


def sample_thought_and_function(
        llm: Any,
        prompt: str,
        template_program: Any,
        *,
        profiler: Any = None,
        operator: str | None = None,
        sample_order: int | None = None,
        stage: str = "generate",
        attach_entire_code: bool = False,
        postprocess: Callable[[Any], Any] | None = None,
):
    response = llm.draw_sample(prompt)
    thought = trim_braced_thought(response)
    code = SampleTrimmer.trim_preface_of_function(response)
    function = SampleTrimmer.sample_to_function(code, template_program)

    if function is not None and attach_entire_code:
        function.entire_code = str(SampleTrimmer.sample_to_program(code, template_program))
    if function is not None and postprocess is not None:
        function = postprocess(function)

    log_sampler_llm_call(
        profiler,
        stage=stage,
        operator=operator,
        sample_order=sample_order,
        prompt=prompt,
        response=response,
        parsed_thought=thought,
        thought_parse_success=thought is not None,
        function_parse_success=function is not None,
    )
    return thought, function

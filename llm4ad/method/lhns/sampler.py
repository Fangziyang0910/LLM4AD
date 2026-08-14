from __future__ import annotations

from typing import Tuple

from .func_ruin import LHNSFunction, LHNSProgram
from .sampling import sample_thought_and_function, trim_braced_thought
from ...base import LLM


class LHNSSampler:
    def __init__(self, sampler: LLM, template_program: str | LHNSProgram, profiler=None):
        self._sampler = sampler
        self._template_program = template_program
        self._profiler = profiler

    def get_thought_and_function(self, prompt: str, *, operator=None, sample_order=None) -> Tuple[str, LHNSFunction]:
        return sample_thought_and_function(
            self._sampler,
            prompt,
            self._template_program,
            profiler=self._profiler,
            operator=operator,
            sample_order=sample_order,
            postprocess=LHNSFunction.convert_function_to_lhnsfunction,
        )

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        return trim_braced_thought(response)

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

from llm4ad.base import Function
from llm4ad.method.mcts_ahd.prompt import MAPrompt


@dataclass(frozen=True, slots=True)
class TraceState:
    """One ordered algorithm state in the MCTS ancestry path."""

    description: str
    score: float | None
    change: str


class MCTSTracePrompt(MAPrompt):
    """MCTS-AHD prompts with ordered trace context for selected operators."""

    @classmethod
    def build_trace_states(cls, indivs: Sequence[Function]) -> tuple[TraceState, ...]:
        states: list[TraceState] = []
        previous_score: float | None = None
        for index, indiv in enumerate(indivs):
            score = getattr(indiv, "score", None)
            if index == 0:
                change = "start"
            elif score is None or previous_score is None:
                change = "unknown"
            elif score > previous_score:
                change = "improved"
            elif score < previous_score:
                change = "regressed"
            else:
                change = "unchanged"
            states.append(
                TraceState(
                    description=getattr(indiv, "algorithm", ""),
                    score=score,
                    change=change,
                )
            )
            previous_score = score
        return tuple(states)

    @classmethod
    def get_prompt_s1_trace(
        cls,
        task_prompt: str,
        trace_indivs: Sequence[Function],
        current_indiv: Function,
        template_function: Function,
    ):
        assert hasattr(current_indiv, "algorithm")

        temp_func = copy.deepcopy(template_function)
        temp_func.body = ""
        trace_prompt = cls.format_trace(cls.build_trace_states(trace_indivs))
        current_prompt = cls.format_current_algorithm(current_indiv)
        prompt_content = f'''{task_prompt}
{trace_prompt}

[Current Algorithm To Improve]
{current_prompt}

Please help me create a new algorithm that is inspired by the ordered trace above with its score higher than any state in the trace.
1. Firstly, list some ideas in the trace states that are clearly helpful to a better algorithm.
2. Secondly, based on the listed ideas and the trace outcomes, describe the design idea and main steps of your new algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m1_trace(
        cls,
        task_prompt: str,
        indi: Function,
        template_function: Function,
        trace_indivs: Sequence[Function],
    ):
        assert hasattr(indi, "algorithm")

        temp_func = copy.deepcopy(template_function)
        temp_func.body = ""
        trace_prompt = cls.format_trace(cls.build_trace_states(trace_indivs))
        prompt_content = f'''{task_prompt}
{trace_prompt}

I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Use the ordered trace above to understand which algorithm states improved or regressed before modifying the current algorithm.
Please create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments, while considering the improvement pattern shown by the trace.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m2_trace(
        cls,
        task_prompt: str,
        indi: Function,
        template_function: Function,
        trace_indivs: Sequence[Function],
    ):
        assert hasattr(indi, "algorithm")

        temp_func = copy.deepcopy(template_function)
        temp_func.body = ""
        trace_prompt = cls.format_trace(cls.build_trace_states(trace_indivs))
        prompt_content = f'''{task_prompt}
{trace_prompt}

I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Use the ordered trace above to understand which parameter settings or equations appear to improve or regress along the current path.
Please identify the main algorithm parameters and help me in creating a new algorithm that has different parameter settings to equations compared to the provided algorithm, while considering the improvement pattern shown by the trace.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def format_trace(cls, states: Sequence[TraceState]) -> str:
        lines = [
            "[Algorithm Improvement Trace]",
            "Higher score is better.",
            "The following states are ordered from earliest to current.",
        ]
        if not states:
            lines.append("No prior trace state is available.")
            return "\n".join(lines)
        for index, state in enumerate(states, start=1):
            lines.extend(
                [
                    "",
                    f"State {index}:",
                    f"Algorithm description: {state.description}",
                    f"Score: {cls._format_score(state.score)}",
                    f"Change from previous: {state.change}",
                ]
            )
        return "\n".join(lines)

    @classmethod
    def format_current_algorithm(cls, indiv: Function) -> str:
        return "\n".join(
            [
                f"Algorithm description: {getattr(indiv, 'algorithm', '')}",
                f"Score: {cls._format_score(cls._score(indiv))}",
                "Code:",
                str(indiv),
            ]
        )

    @staticmethod
    def _score(indiv: Function) -> float | None:
        return getattr(indiv, "score", None)

    @staticmethod
    def _format_score(score: float | None) -> str:
        if score is None:
            return "unknown"
        return f"{score:.6g}"

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
    operator: str
    score_delta: float | None


class MCTSTracePrompt(MAPrompt):
    """MCTS-AHD prompts with ordered trace context for all expansion operators."""

    MAX_TRACE_STATES = 10

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
                    operator=getattr(indiv, "operator", "unknown") or "unknown",
                    score_delta=None if score is None or previous_score is None else score - previous_score,
                )
            )
            previous_score = score
        return tuple(states)

    @classmethod
    def select_trace_states(cls, states: Sequence[TraceState]) -> tuple[TraceState, ...]:
        if len(states) <= cls.MAX_TRACE_STATES:
            return tuple(states)
        return tuple([states[0], *states[-(cls.MAX_TRACE_STATES - 1):]])

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
        trace_prompt = cls.format_trace(
            cls.build_trace_states(trace_indivs),
            focus="s1 synthesizes useful parts from the current ancestry path into one new candidate.",
            guidance_lines=(
                "Look for ideas that survived several steps, ideas that became worse after being combined, and changes that may complement each other.",
                "The new algorithm should synthesize the current path; it should not merely copy the highest-scoring ancestor or undo everything after a regression.",
            ),
        )
        current_prompt = cls.format_current_algorithm(current_indiv)
        prompt_content = f'''{task_prompt}
{trace_prompt}

[Current Algorithm To Improve]
{current_prompt}

Please help me create a new algorithm by using s1-style synthesis on the current ancestry path.
1. Firstly, identify which path-level ideas should be combined, simplified, or left out for this synthesis.
2. Secondly, based on those path-level clues, describe the design idea and main steps of your new algorithm in one sentence. The description must be inside within boxed {{}}.
3. Thirdly, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e1_trace(cls, task_prompt: str, indivs: Sequence[Function], template_function: Function):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        evidence_prompt = cls.format_root_evidence(indivs)
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} branch algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}\nObjective value: {str(-indi.score)}\n'
        return f'''{evidence_prompt}

{task_prompt}
I have {len(indivs)} existing branch algorithms with their codes as follows:
{indivs_prompt}
For e1, create a new root-level exploration branch with a different form from the existing branch algorithms.
Use the branch evidence to avoid repeating already-covered idea families and to choose a genuinely different structure, flow, or algorithmic mechanism.
The branch scores are only search evidence; they should help you judge coverage and risk, not decide that the best-looking branch must be copied.
1. First, describe the design idea and main steps of your new exploratory branch in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''

    @classmethod
    def get_prompt_e2_trace(
        cls,
        task_prompt: str,
        indivs: Sequence[Function],
        template_function: Function,
        trace_indivs: Sequence[Function],
    ):
        for indi in indivs:
            assert hasattr(indi, 'algorithm')

        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        trace_prompt = cls.format_trace(
            cls.build_trace_states(trace_indivs),
            focus="e2 improves the current branch by borrowing from an elite reference while keeping the current branch as the base form.",
            guidance_lines=(
                "Use the current branch trace to decide which local components are stable enough to preserve and which recent changes need revision.",
                "Borrow from the elite reference only where it resolves a weakness suggested by the current branch trace; do not convert the candidate into a copy of the elite.",
            ),
        )
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}\nObjective value: {str(-indi.score)}\n'

        prompt_content = f'''{task_prompt}
{trace_prompt}

I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please create a new algorithm that has a similar form to the No.{len(indivs)} algorithm and is inspired by the No.{1} algorithm. The new algorithm should have a objective value lower than both algorithms.
For e2, treat No.1 as the elite reference and No.{len(indivs)} as the current branch base.
1. Firstly, identify where the elite reference can improve the current branch according to the current branch trace.
2. Secondly, based on this operator-specific evidence, describe the design idea based on the No.{len(indivs)} algorithm and main steps of your algorithm in one sentence. The description must be inside within boxed {{}}.
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
        trace_prompt = cls.format_trace(
            cls.build_trace_states(trace_indivs),
            focus="m1 performs a structural mutation of the current algorithm.",
            guidance_lines=(
                "Use regressions, unchanged steps, or repeated operator outcomes as clues for where the current structure may be saturated.",
                "Prefer a different mechanism, control flow, or equation family; do not make this only a small parameter retuning.",
            ),
        )
        prompt_content = f'''{task_prompt}
{trace_prompt}

I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments.
1. First, describe the structural mutation suggested by the trace and then describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
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
        trace_prompt = cls.format_trace(
            cls.build_trace_states(trace_indivs),
            focus="m2 performs a parameter, weighting, threshold, or equation-level mutation of the current algorithm.",
            guidance_lines=(
                "Use score deltas to judge which kinds of local tuning have already been tried and whether the current formula may be too aggressive or too weak.",
                "Keep the main algorithmic structure recognizable; focus on parameterization, scoring equations, thresholds, exponents, or normalization terms.",
            ),
        )
        prompt_content = f'''{task_prompt}
{trace_prompt}

I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please identify the main algorithm parameters and help me create a new algorithm with different parameter settings or equations compared to the provided algorithm.
1. First, describe the parameter or equation-level mutation suggested by the trace and then describe your new algorithm and main steps in one sentence. The description must be inside within boxed {{}}.
2. Next, implement the idea in the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def format_trace(
        cls,
        states: Sequence[TraceState],
        *,
        focus: str | None = None,
        guidance_lines: Sequence[str] = (),
    ) -> str:
        selected_states = cls.select_trace_states(states)
        lines = [
            "[Algorithm Improvement Trace]",
            "Higher score is better.",
            "The trace records search history for the current ancestry path; it is not a list of code to copy.",
            "The following states are ordered from earliest to current within the shown subset.",
        ]
        if focus is not None:
            lines.extend(["", f"Trace focus for this operator: {focus}"])
        if guidance_lines:
            lines.extend(["", "[Operator-Specific Trace Use]"])
            lines.extend(f"- {line}" for line in guidance_lines)
        if not states:
            lines.append("No prior trace state is available.")
            return "\n".join(lines)
        if len(states) != len(selected_states):
            lines.append(
                f"Showing {len(selected_states)} of {len(states)} trace states: the first state and the most recent {len(selected_states) - 1} states."
            )
        for index, state in enumerate(selected_states, start=1):
            lines.extend(
                [
                    "",
                    f"State {index}:",
                    f"Algorithm description: {state.description}",
                    f"Score: {cls._format_score(state.score)}",
                    f"Change from previous full-trace state: {state.change}",
                    f"Score delta from previous full-trace state: {cls._format_delta(state.score_delta)}",
                    f"Operator that produced this state: {state.operator}",
                ]
            )
        return "\n".join(lines)

    @classmethod
    def format_root_evidence(cls, indivs: Sequence[Function]) -> str:
        lines = [
            "[Search Evidence From Existing Branches]",
            "No ordered ancestry trace is available for e1 because e1 creates a new root-level exploration branch.",
            "For e1, use these branch algorithms to understand which idea families have already been covered.",
            "The goal is exploration diversity: create a different branch, not a local refinement of the best-looking branch.",
        ]
        if not indivs:
            lines.append("No branch evidence is available yet.")
            return "\n".join(lines)
        for index, indiv in enumerate(indivs, start=1):
            lines.extend(
                [
                    "",
                    f"Branch evidence {index}:",
                    f"Algorithm description: {getattr(indiv, 'algorithm', '')}",
                    f"Score: {cls._format_score(cls._score(indiv))}",
                    f"Operator that produced this branch state: {getattr(indiv, 'operator', 'unknown') or 'unknown'}",
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

    @staticmethod
    def _format_delta(delta: float | None) -> str:
        if delta is None:
            return "unknown"
        return f"{delta:+.6g}"

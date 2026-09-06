from __future__ import annotations

import pytest

from llm4ad.base import Evaluation
from llm4ad.method.traceaad_v10_3.schema import Node
from llm4ad.method.traceaad_v10_4.prompts import (
    build_code_prompt,
    build_idea_prompt,
    render_formation_history,
)
from llm4ad.method.traceaad_v10_4.traceaad import TraceAADV104


class _Evaluation(Evaluation):
    def __init__(self):
        super().__init__(
            template_program="def score(x):\n    pass",
            task_description="Return a numeric score.",
            safe_evaluate=False,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(1)


class _LLM:
    def __init__(self, *responses: str | Exception):
        self.calls: list[tuple[str, dict]] = []
        self.responses = iter(responses)

    def draw_sample(self, prompt: str, **kwargs) -> str:
        self.calls.append((prompt, kwargs))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def _method(tmp_path, llm) -> TraceAADV104:
    return TraceAADV104(
        evaluation=_Evaluation(),
        llm=llm,
        run_dir=tmp_path,
        budget=10,
        n_roots=1,
        idea_output_tokens=1024,
    )


def test_v104_history_includes_the_current_formation_event() -> None:
    root = Node(0, "root", "root idea", 1.0, parent_id=None)
    middle = Node(1, "middle", "middle idea", 3.0, parent_id=0)
    current = Node(2, "current", "current idea", 2.0, parent_id=1)

    history = render_formation_history(current, [middle, root], max_events=8)

    assert "Idea: middle idea" in history
    assert "Result: improve (Fitness: 1.0 -> 3.0)" in history
    assert "Idea: current idea" in history
    assert "Result: regress (Fitness: 3.0 -> 2.0)" in history


def test_v104_idea_and_code_prompts_have_distinct_contexts() -> None:
    root = Node(0, "def score(x):\n    return 1", "root", 1.0)
    current = Node(
        1,
        "def score(x):\n    return 2",
        "current",
        2.0,
        parent_id=0,
    )
    donor = Node(2, "def score(x):\n    return 3", "donor", 3.0)
    idea_prompt = build_idea_prompt(
        task_contract="TASK",
        current=current,
        ancestors=[root],
        operator="Fuse",
        donor=donor,
        max_events=8,
    )
    code_prompt = build_code_prompt(
        task_contract="TASK",
        current=current,
        donor=donor,
        idea="combine their useful geometry",
    )

    assert "# Recent Algorithm Improvement History" in idea_prompt
    assert "identify the strengths" in idea_prompt
    assert "not the Python implementation" in idea_prompt
    assert "donor" in idea_prompt
    assert "# Recent Algorithm Improvement History" not in code_prompt
    assert "# Design Direction" not in code_prompt
    assert "Operator:" not in code_prompt
    assert "combine their useful geometry" in code_prompt
    assert "donor" in code_prompt


def test_v104_uses_one_idea_call_then_one_code_call(tmp_path) -> None:
    idea = "Use a two-stage geometric estimate.\nThen rank candidates by its bound."
    llm = _LLM(idea, "```python\ndef score(x):\n    return 7\n```")
    method = _method(tmp_path, llm)

    attempt = method._generate("Init", None, [], None)
    method._evaluate(attempt)

    assert attempt.idea == idea
    assert attempt.fitness == 7.0
    assert method.budget_used == 1
    assert (attempt.idea_calls, attempt.code_calls) == (1, 1)
    assert llm.calls[0][1]["max_tokens"] == 1024
    assert llm.calls[1][1]["max_tokens"] == 16384
    assert idea in llm.calls[1][0]
    assert "# Recent Algorithm Improvement History" not in llm.calls[1][0]


def test_v104_resumes_saved_idea_without_generating_another(tmp_path) -> None:
    first = _LLM("A durable design", RuntimeError("connection lost"))
    method = _method(tmp_path, first)
    with pytest.raises(RuntimeError, match="connection lost"):
        method._generate("Init", None, [], None)

    second = _LLM("```python\ndef score(x):\n    return 9\n```")
    resumed = _method(tmp_path, second)
    resumed._load_state()
    attempt = resumed._generate("Init", None, [], None)

    assert attempt.idea == "A durable design"
    assert (attempt.idea_calls, attempt.code_calls) == (0, 1)
    assert len(second.calls) == 1
    assert "A durable design" in second.calls[0][0]

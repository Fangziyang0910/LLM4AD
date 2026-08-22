"""Protocol tests for the V9.15 slot budget and curated error feedback."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from llm4ad.base import Evaluation, LLM, SecureEvaluator
from llm4ad.method.traceaad_v9_15 import RunArtifacts
from llm4ad.method.traceaad_v9_15.prompt import format_failure_feedback
from llm4ad.method.traceaad_v9_15 import TraceAADV915

TEMPLATE = """def choose(value: int) -> int:
    return value
"""

RAISING_CODE = """def choose(value: int) -> int:
    return value + None
"""


class ScriptedLLM(LLM):
    """Returns scripted programs and records every prompt it receives."""

    def __init__(
        self, responses: list[str], *, invalid_at: set[int] | None = None
    ) -> None:
        super().__init__()
        self.responses = responses
        self.invalid_at = invalid_at or set()
        self.calls = 0
        self.prompts: list[str] = []

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.prompts.append(prompt)
        self.calls += 1
        if self.calls in self.invalid_at:
            return "unusable response"
        return self.responses[self.calls - 1]


class IncrementalEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            safe_evaluate=False,
            timeout_seconds=5,
        )

    def evaluate_program(self, program_str: str, callable_func, **kwargs) -> float | None:
        return None if callable_func is None else float(callable_func(10))


def _program(value: float) -> str:
    return (
        f"Idea: candidate {value}\n"
        "Code:\n```python\n"
        "def choose(value: int) -> int:\n"
        f"    return {value}\n"
        "```"
    )


def test_exhausted_repairs_consume_exactly_one_slot(tmp_path: Path) -> None:
    run_dir = tmp_path / "exhausted"
    method = TraceAADV915(
        llm=ScriptedLLM(
            [_program(10.0), _program(11.0), _program(12.0), _program(13.0)],
            invalid_at={2, 3, 4},
        ),
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=2,
        n_roots=1,
    )
    method.run()

    assert method._n_eval == 2  # both slots charged on their initial attempts
    assert method._n_calls == 4  # two extra repair calls were free of budget
    assert len(method._tree.valid_algorithms()) == 1  # only the root survived
    with (run_dir / "evaluations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["attempt_kind"] for row in rows] == [
        "initial",
        "initial",
        "repair",
        "repair",
    ]
    assert [row["attempt"] for row in rows] == ["1", "1", "2", "3"]
    assert [row["status"] for row in rows] == ["ok", "exec_error", "exec_error", "exec_error"]
    summary = json.loads((run_dir / "logs/summary.json").read_text())
    assert summary["status"] == "finished"
    assert summary["budget_slots"] == 2
    assert summary["evaluator_call_count"] == 4


def test_repair_prompt_carries_type_traceback_and_candidate_frames(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "feedback"
    llm = ScriptedLLM(
        [
            _program(10.0),
            f"Idea: broken\nCode:\n```python\n{RAISING_CODE}```",
            _program(14.0),
            _program(15.0),
        ],
    )
    method = TraceAADV915(
        llm=llm,
        evaluation=IncrementalEvaluation(),
        artifacts=RunArtifacts(run_dir, console_output=False),
        budget=3,
        n_roots=1,
    )
    method.run()

    repair_prompt = llm.prompts[2]
    assert "Failure during" in repair_prompt
    assert "TypeError:" in repair_prompt
    assert 'File "<string>"' in repair_prompt
    assert "candidate frames" in repair_prompt
    assert "llm4ad" not in repair_prompt  # evaluator internals are filtered out
    assert method._n_eval == 3
    assert method._n_calls == 4  # initial failure + one successful repair


def test_failure_feedback_keeps_message_and_adds_timeout_guidance() -> None:
    traceback_text = (
        'Traceback (most recent call last):\n'
        '  File "/opt/llm4ad/base/evaluate.py", line 1, in _evaluate\n'
        '    res = self._evaluator.evaluate_program(program_str, program_callable)\n'
        '  File "<string>", line 2, in choose\n'
        '    return value + None\n'
        'TypeError: unsupported operand type(s) for +: \'int\' and \'NoneType\''
    )
    feedback = format_failure_feedback(
        error_type="TypeError",
        error="unsupported operand type(s) for +: 'int' and 'NoneType'",
        traceback=traceback_text,
    )
    assert feedback.startswith("TypeError: unsupported operand")
    assert 'File "<string>", line 2, in choose' in feedback
    assert "evaluate.py" not in feedback
    assert "TimeoutError" not in feedback

    long_message = "x" * 5000
    feedback = format_failure_feedback(
        error_type="ValueError", error=long_message, traceback=None
    )
    assert long_message in feedback  # the message itself is never truncated

    timeout = format_failure_feedback(
        error_type="TimeoutError", error="evaluation exceeded 20s", traceback=None
    )
    assert "time limit" in timeout


def test_evaluator_exposes_traceback_for_failed_programs() -> None:
    evaluator = SecureEvaluator(IncrementalEvaluation())
    outcome = evaluator.evaluate_program_with_details(RAISING_CODE)

    assert outcome.failure_kind == "runtime_error"
    assert outcome.error_type == "TypeError"
    assert outcome.traceback is not None
    assert "<string>" in outcome.traceback

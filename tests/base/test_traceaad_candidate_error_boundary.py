"""Candidate error-boundary contract tests for TraceAAD V9.7.

A completed model response is leniently extracted (fenced block, else text
after ``Code:``, else the whole response), always reaches the evaluator,
consumes one unit of evaluation budget, and failures record the evaluator's
real failure kind instead of a candidate-level parse state.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_7 import TraceAADV97

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedLLM(LLM):
    """Return one fixed response per prompt; optionally vary code by call count."""

    def __init__(self, response: str, *, increment: bool = False) -> None:
        super().__init__()
        self.response = response
        self.increment = increment
        self.calls = 0

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.calls += 1
        if self.increment:
            return fenced(
                "def choose(value: int) -> int:\n"
                f"    return value + {self.calls}",
                f"candidate {self.calls}",
            )
        return self.response

    def count_tokens(self, text: str) -> int:
        return len(text) // 4 + 1


class TransportErrorLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.calls += 1
        raise ConnectionError("connection refused")

    def count_tokens(self, text: str) -> int:
        return len(text) // 4 + 1


class RawEvaluation(Evaluation):
    """Return the raw callable result so NaN/Inf/non-numeric reach the method."""

    def __init__(self, timeout_seconds: int | float = 2) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            timeout_seconds=timeout_seconds,
        )

    def evaluate_program(self, program_str: str, callable_func, **kwargs):
        return callable_func(3)


def fenced(body: str, idea: str | None = None) -> str:
    prefix = "" if idea is None else f"Idea: {idea}\n"
    return f"{prefix}Code:\n```python\n{body}\n```"


def code_marked(body: str, idea: str | None = None) -> str:
    prefix = "" if idea is None else f"Idea: {idea}\n"
    return f"{prefix}Code:\n{body}"


def build_method(
    llm: LLM,
    evaluation: Evaluation,
    run_dir: Path,
    *,
    budget: int,
    n_init: int = 1,
    resume_from: Path | None = None,
) -> TraceAADV97:
    run_dir.mkdir(parents=True, exist_ok=True)
    from llm4ad.method.traceaad_v9_7 import RunArtifacts

    return TraceAADV97(
        llm,
        evaluation,
        RunArtifacts(run_dir, console_output=False),
        budget,
        n_roots=n_init,
        checkpoint_dir=run_dir / "checkpoints",
        seed=1,
        resume_from=None if resume_from is None else Path(resume_from),
    )


def run_one(response: str, tmp_path: Path, *, budget: int = 1, increment: bool = False):
    llm = ScriptedLLM(response, increment=increment)
    method = build_method(llm, RawEvaluation(), tmp_path / "v9_7", budget=budget)
    method.run()
    return method, llm


def valid_state_count(method) -> int:
    return len(method._forest.programs())


def candidate_rows(tmp_path: Path) -> list[dict]:
    path = tmp_path / "v9_7" / "evaluations.csv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def statuses(rows: list[dict]) -> list[str]:
    return [row["status"] for row in rows]


# ==============================================================================
# 1-2. Lenient extraction: fenced block and explicit Code: marker both succeed
# ==============================================================================


def test_fenced_response_creates_valid_search_state(tmp_path: Path) -> None:
    method, _ = run_one("", tmp_path, increment=True)
    assert method._n_eval == 1
    assert valid_state_count(method) == 1
    assert statuses(candidate_rows(tmp_path)) == ["ok"]


def test_code_marker_response_creates_valid_search_state(tmp_path: Path) -> None:
    class CodeMarkerLLM(ScriptedLLM):
        def draw_sample(self, prompt: str, *args, **kwargs) -> str:
            self.calls += 1
            return code_marked(
                "def choose(value: int) -> int:\n"
                f"    return value + {self.calls + 10}",
                "add a constant",
            )

    llm = CodeMarkerLLM("")
    method = build_method(llm, RawEvaluation(), tmp_path / "v9_7", budget=1)
    method.run()
    assert method._n_eval == 1
    assert valid_state_count(method) == 1
    assert statuses(candidate_rows(tmp_path)) == ["ok"]


# ==============================================================================
# 3-8. Failures consume budget, record real kinds, and create no search state
# ==============================================================================

FAILURE_RESPONSES = {
    "plain_text": ("This response has no code at all, only discussion.", "exec_error"),
    "syntax_error": (
        fenced("def choose(value: int) -> int:\n    return value +"),
        "exec_error",
    ),
    "runtime_error": (
        fenced("def choose(value: int) -> int:\n    raise ValueError('boom')"),
        "runtime_error",
    ),
}


@pytest.mark.parametrize("case", list(FAILURE_RESPONSES))
def test_invalid_candidates_enter_evaluator_and_record_real_kind(
    case: str, tmp_path: Path
) -> None:
    response, expected_status = FAILURE_RESPONSES[case]
    method, _ = run_one(response, tmp_path)
    assert method._n_eval == 1
    assert valid_state_count(method) == 0
    assert statuses(candidate_rows(tmp_path)) == [expected_status]


def test_timeout_records_timeout_and_consumes_budget(tmp_path: Path) -> None:
    response = fenced(
        "import time\n\n\ndef choose(value: int) -> int:\n"
        "    time.sleep(5)\n    return value"
    )
    llm = ScriptedLLM(response)
    method = build_method(
        llm, RawEvaluation(timeout_seconds=1), tmp_path / "v9_7", budget=1
    )
    method.run()
    assert method._n_eval == 1
    assert valid_state_count(method) == 0
    assert statuses(candidate_rows(tmp_path)) == ["timeout"]


@pytest.mark.parametrize(
    "body",
    [
        "def choose(value: int) -> int:\n    return None",
        "def choose(value: int) -> int:\n    return float('nan')",
        "def choose(value: int) -> int:\n    return float('inf')",
        "def choose(value: int) -> int:\n    return 'not-a-number'",
    ],
    ids=["none", "nan", "inf", "non-numeric"],
)
def test_non_finite_results_record_invalid_result(body: str, tmp_path: Path) -> None:
    method, _ = run_one(fenced(body), tmp_path)
    assert method._n_eval == 1
    assert valid_state_count(method) == 0
    assert statuses(candidate_rows(tmp_path)) == ["invalid_result"]


# ==============================================================================
# 9. Initialization terminates by budget exhaustion under persistent failures
# ==============================================================================


def test_init_stops_by_budget_exhaustion_with_invalid_candidates(tmp_path: Path) -> None:
    llm = ScriptedLLM("no code here, just prose")
    method = build_method(
        llm, RawEvaluation(), tmp_path / "v9_7", budget=3, n_init=5
    )
    method.run()
    assert method._n_eval == 3
    assert valid_state_count(method) == 0
    assert llm.calls == 3  # one LLM response per consumed evaluation
    assert statuses(candidate_rows(tmp_path)) == ["exec_error"] * 3


# ==============================================================================
# 10. Pending checkpoint responses are evaluated exactly once after resume
# ==============================================================================


def test_pending_checkpoint_resume_evaluates_once(tmp_path: Path) -> None:
    import importlib

    schema = importlib.import_module("llm4ad.method.traceaad_v9_7.schema")
    checkpoint_module = importlib.import_module("llm4ad.method.traceaad_v9_7.checkpoint")

    run_dir = tmp_path / "v9_7"
    evaluation = RawEvaluation()
    method = build_method(
        ScriptedLLM("", increment=True),
        evaluation,
        run_dir,
        budget=1,
    )
    program = method._forest.add_program(
        code="def choose(value: int) -> int:\n    return 0", fitness=0.0, order=1
    )
    child = method._forest.add_root(program_id=program.id, order=1)
    method._forest.get_anchor(child.id).n = 1
    method._initialization_complete = True
    method._n_eval = 1
    method._pending = schema.Pending(
        id=method._forest.next_attempt_id(),
        anchor_id=child.id,
        stage="search",
        iteration=0,
        order=2,
        intent="refine",
        response=fenced("def choose(value: int) -> int:\n    return 99", "resume me"),
    )
    checkpoint_module.save_checkpoint(method)
    del method

    resumed_llm = ScriptedLLM("", increment=True)
    resumed = build_method(
        resumed_llm,
        evaluation,
        run_dir,
        budget=1,
        resume_from=run_dir / "checkpoints" / "latest.json",
    )
    assert resumed._pending is not None
    resumed.run()
    assert resumed._n_eval == 2
    assert resumed_llm.calls == 0  # pending response evaluated, never redrawn
    assert resumed._pending is None
    assert valid_state_count(resumed) == 2


# ==============================================================================
# 11. Transport errors: no budget consumed, run fails
# ==============================================================================


def test_transport_errors_are_immediately_visible(tmp_path: Path) -> None:
    llm = TransportErrorLLM()
    method = build_method(llm, RawEvaluation(), tmp_path / "v9_7", budget=1)
    with pytest.raises(ConnectionError, match="connection refused"):
        method.run()
    assert method._n_eval == 0
    assert valid_state_count(method) == 0
    assert candidate_rows(tmp_path) == []

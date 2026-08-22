"""Unified candidate error-boundary contract tests for TraceAAD V4 through V9.7.

Every version must follow the V9.14 semantics: a completed model response is
leniently extracted (fenced block, else text after ``Code:``, else the whole
response), always reaches the evaluator, consumes one unit of evaluation
budget, and failures record the evaluator's real failure kind instead of a
candidate-level parse state.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v4 import TraceAADV4
from llm4ad.method.traceaad_v5 import TraceAADV5
from llm4ad.method.traceaad_v8 import TraceAADV8
from llm4ad.method.traceaad_v9 import TraceAADV9
from llm4ad.method.traceaad_v9_7 import TraceAADV97

TEMPLATE = """def choose(value: int) -> int:
    return value
"""

VERSIONS = [
    "v4",
    "v5",
    "v8",
    "v9",
    "v9_7",
]


class ScriptedLLM(LLM):
    """Return one fixed response per prompt; optionally vary code by call count.

    ``increment=True`` makes every response a fresh valid program, which keeps
    the dedup paths of the forest versions from looping on identical code.
    """

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


def _artifacts(version: str, run_dir: Path):
    module = {
        "v4": "traceaad_v4",
        "v5": "traceaad_v5",
        "v8": "traceaad_v8",
        "v9": "traceaad_v9",
    }.get(version, f"traceaad_{version}")
    import importlib

    artifacts_type = importlib.import_module(f"llm4ad.method.{module}").RunArtifacts
    try:
        return artifacts_type(run_dir, console_output=False)
    except TypeError:
        return artifacts_type(run_dir)


def build_method(
    version: str,
    llm: LLM,
    evaluation: Evaluation,
    run_dir: Path,
    *,
    budget: int,
    n_init: int = 1,
    resume_from: Path | None = None,
):
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _artifacts(version, run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    resume = None if resume_from is None else Path(resume_from)
    if version == "v4":
        return TraceAADV4(
            llm,
            evaluation,
            artifacts,
            budget,
            n_init=n_init,
            checkpoint_dir=checkpoint_dir,
            resume_from=resume,
        )
    if version == "v5":
        return TraceAADV5(
            llm,
            evaluation,
            artifacts,
            budget,
            n_init=n_init,
            checkpoint_dir=checkpoint_dir,
            resume_from=resume,
        )
    if version == "v8":
        return TraceAADV8(
            llm,
            evaluation,
            artifacts,
            budget,
            n_init=n_init,
            context_token_limit=32768,
            checkpoint_dir=checkpoint_dir,
            resume_from=resume,
        )
    if version == "v9":
        return TraceAADV9(
            llm,
            evaluation,
            artifacts,
            budget,
            n_init=n_init,
            context_token_limit=32768,
            checkpoint_dir=checkpoint_dir,
            resume_from=resume,
        )
    family_kwargs = {
        "llm": llm,
        "evaluation": evaluation,
        "artifacts": artifacts,
        "budget": budget,
        "n_roots": n_init,
        "checkpoint_dir": checkpoint_dir,
        "seed": 1,
        "resume_from": resume,
    }
    return TraceAADV97(**family_kwargs)


def run_one(version: str, response: str, tmp_path: Path, *, budget: int = 1, increment: bool = False):
    llm = ScriptedLLM(response, increment=increment)
    method = build_method(version, llm, RawEvaluation(), tmp_path / version, budget=budget)
    method.run()
    return method, llm


def eval_count(method) -> int:
    counter = getattr(method, "_n_eval", None)
    if counter is not None:
        return counter
    return method._tot_sample_nums


def valid_state_count(method, version: str) -> int:
    if version in {"v4", "v5"}:
        return len(method._graph.nodes())
    if version in {"v8", "v9"}:
        return len(method._tree.root.child_ids)
    return len(method._forest.programs())


def candidate_rows(version: str, run_dir: Path) -> list[dict]:
    if version in {"v4", "v5", "v8", "v9"}:
        path = run_dir / version / "artifacts" / "candidates.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]
    path = run_dir / version / "evaluations.csv"
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def statuses(rows: list[dict]) -> list[str]:
    return [row["status"] for row in rows]


# ==============================================================================
# 1-2. Lenient extraction: fenced block and explicit Code: marker both succeed
# ==============================================================================

@pytest.mark.parametrize("version", VERSIONS)
def test_fenced_response_creates_valid_search_state(version: str, tmp_path: Path) -> None:
    method, _ = run_one(version, "", tmp_path, increment=True)
    assert eval_count(method) == 1
    assert valid_state_count(method, version) == 1
    assert statuses(candidate_rows(version, tmp_path)) == ["ok"]


@pytest.mark.parametrize("version", VERSIONS)
def test_code_marker_response_creates_valid_search_state(
    version: str, tmp_path: Path
) -> None:
    class CodeMarkerLLM(ScriptedLLM):
        def draw_sample(self, prompt: str, *args, **kwargs) -> str:
            self.calls += 1
            return code_marked(
                "def choose(value: int) -> int:\n"
                f"    return value + {self.calls + 10}",
                "add a constant",
            )

    llm = CodeMarkerLLM("")
    method = build_method(version, llm, RawEvaluation(), tmp_path / version, budget=1)
    method.run()
    assert eval_count(method) == 1
    assert valid_state_count(method, version) == 1
    assert statuses(candidate_rows(version, tmp_path)) == ["ok"]


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


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("case", list(FAILURE_RESPONSES))
def test_invalid_candidates_enter_evaluator_and_record_real_kind(
    version: str, case: str, tmp_path: Path
) -> None:
    response, expected_status = FAILURE_RESPONSES[case]
    method, _ = run_one(version, response, tmp_path)
    assert eval_count(method) == 1
    assert valid_state_count(method, version) == 0
    assert statuses(candidate_rows(version, tmp_path)) == [expected_status]


@pytest.mark.parametrize("version", VERSIONS)
def test_timeout_records_timeout_and_consumes_budget(version: str, tmp_path: Path) -> None:
    response = fenced(
        "import time\n\n\ndef choose(value: int) -> int:\n"
        "    time.sleep(5)\n    return value"
    )
    llm = ScriptedLLM(response)
    method = build_method(
        version, llm, RawEvaluation(timeout_seconds=1), tmp_path / version, budget=1
    )
    method.run()
    assert eval_count(method) == 1
    assert valid_state_count(method, version) == 0
    assert statuses(candidate_rows(version, tmp_path)) == ["timeout"]


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
@pytest.mark.parametrize("version", VERSIONS)
def test_non_finite_results_record_invalid_result(
    version: str, body: str, tmp_path: Path
) -> None:
    method, _ = run_one(version, fenced(body), tmp_path)
    assert eval_count(method) == 1
    assert valid_state_count(method, version) == 0
    assert statuses(candidate_rows(version, tmp_path)) == ["invalid_result"]


# ==============================================================================
# 9. Initialization terminates by budget exhaustion under persistent failures
# ==============================================================================

@pytest.mark.parametrize("version", VERSIONS)
def test_init_stops_by_budget_exhaustion_with_invalid_candidates(
    version: str, tmp_path: Path
) -> None:
    llm = ScriptedLLM("no code here, just prose")
    method = build_method(
        version, llm, RawEvaluation(), tmp_path / version, budget=3, n_init=5
    )
    method.run()
    assert eval_count(method) == 3
    assert valid_state_count(method, version) == 0
    assert llm.calls == 3  # one LLM response per consumed evaluation
    assert statuses(candidate_rows(version, tmp_path)) == ["exec_error"] * 3


# ==============================================================================
# 10. Pending checkpoint responses are evaluated exactly once after resume
# ==============================================================================

PENDING_SIMPLE_VERSIONS = ["v9_7"]


@pytest.mark.parametrize("version", PENDING_SIMPLE_VERSIONS)
def test_pending_checkpoint_resume_evaluates_once(version: str, tmp_path: Path) -> None:
    import importlib

    package = f"llm4ad.method.traceaad_{version}"
    schema = importlib.import_module(f"{package}.schema")
    checkpoint_module = importlib.import_module(f"{package}.checkpoint")

    run_dir = tmp_path / version
    evaluation = RawEvaluation()
    method = build_method(
        version,
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
        version,
        resumed_llm,
        evaluation,
        run_dir,
        budget=1,
        resume_from=run_dir / "checkpoints" / "latest.json",
    )
    assert resumed._pending is not None
    resumed.run()
    assert eval_count(resumed) == 2
    assert resumed_llm.calls == 0  # pending response evaluated, never redrawn
    assert resumed._pending is None
    assert valid_state_count(resumed, version) == 2


# ==============================================================================
# 11. Transport errors: no budget consumed, run fails
# ==============================================================================

# All versions draw once and let the transport exception propagate directly.
TRANSPORT_DIRECT_VERSIONS = [
    "v4",
    "v5",
    "v8",
    "v9",
    "v9_7",
]


@pytest.mark.parametrize("version", TRANSPORT_DIRECT_VERSIONS)
def test_transport_errors_are_immediately_visible(version: str, tmp_path: Path) -> None:
    llm = TransportErrorLLM()
    method = build_method(version, llm, RawEvaluation(), tmp_path / version, budget=1)
    with pytest.raises(ConnectionError, match="connection refused"):
        method.run()
    assert eval_count(method) == 0
    assert valid_state_count(method, version) == 0
    assert candidate_rows(version, tmp_path) == []

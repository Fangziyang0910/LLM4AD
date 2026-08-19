from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_11 import (
    CHECKPOINT_VERSION,
    MIN_EXPLORE_REMAINING_EVALS,
    PROTOCOL_ID,
    STAGNATION_WINDOW,
    RunArtifacts,
    TraceAADV911,
)
from llm4ad.method.traceaad_v9_11.checkpoint import dump_state, load_state
from llm4ad.method.traceaad_v9_11.schema import Regime
from llm4ad.method.traceaad_v9_11.traceaad import decide_regime


TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedLLM(LLM):
    def __init__(self, start: int = 0) -> None:
        super().__init__()
        self.calls = start

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        return (
            f"Idea: candidate {self.calls}\n"
            "Code:\n```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {self.calls}\n"
            "```"
        )

    def count_tokens(self, text: str) -> int:
        return len(text) // 4 + 1


class ConstantEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            use_numba_accelerate=False,
            safe_evaluate=False,
            timeout_seconds=10,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return 1.0


def test_regime_clock_uses_completed_responses_and_landing_priority() -> None:
    assert (
        decide_regime(
            completed_responses=STAGNATION_WINDOW - 1,
            last_progress_order=0,
            last_explore_order=0,
            landing_anchor_id=None,
            remaining_evals=100,
        )
        is Regime.DEVELOP
    )
    assert (
        decide_regime(
            completed_responses=STAGNATION_WINDOW,
            last_progress_order=0,
            last_explore_order=0,
            landing_anchor_id=None,
            remaining_evals=100,
        )
        is Regime.EXPLORE
    )
    assert (
        decide_regime(
            completed_responses=STAGNATION_WINDOW,
            last_progress_order=0,
            last_explore_order=0,
            landing_anchor_id=42,
            remaining_evals=100,
        )
        is Regime.LANDING
    )
    assert (
        decide_regime(
            completed_responses=STAGNATION_WINDOW,
            last_progress_order=0,
            last_explore_order=0,
            landing_anchor_id=None,
            remaining_evals=MIN_EXPLORE_REMAINING_EVALS - 1,
        )
        is Regime.DEVELOP
    )


def test_v911_smoke_runs_develop_explore_and_one_landing(tmp_path: Path) -> None:
    artifacts = RunArtifacts(tmp_path, console_output=False)
    method = TraceAADV911(
        llm=ScriptedLLM(),
        evaluation=ConstantEvaluation(),
        artifacts=artifacts,
        budget=30,
        context_limit=32768,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=3,
    )
    method.run()

    summary = json.loads((tmp_path / "logs" / "summary.json").read_text())
    assert summary["status"] == "finished"
    assert summary["evaluator_call_count"] == 30
    assert summary["n_roots"] == 8
    assert summary["n_iterations"] == 14
    assert summary["n_develop"] == 12
    assert summary["n_explore"] == 1
    assert summary["n_landing"] == 1
    assert summary["n_valid_explore_children"] == 1
    assert summary["last_explore_order"] == 9
    assert summary["landing_anchor_id"] is None
    programs = method._forest.programs()
    assert all(len(program.code_hash) == 64 for program in programs)
    assert all(program.evaluation_seconds is not None for program in programs)

    events = [
        json.loads(line)
        for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines()
    ]
    selected = [item for item in events if item["event"] == "regime_selected"]
    completed = [item for item in events if item["event"] == "regime_completed"]
    assert [item["regime"] for item in selected] == (
        ["develop"] * 8 + ["explore", "landing"] + ["develop"] * 4
    )
    assert [item["regime"] for item in completed] == [
        item["regime"] for item in selected
    ]
    assert selected[8]["landing_override"] is False
    assert selected[9]["landing_override"] is True
    assert selected[9]["selected_state_id"] == completed[8]["child_id"]


def test_v911_checkpoint_roundtrip_preserves_regime_state(tmp_path: Path) -> None:
    source = TraceAADV911(
        llm=ScriptedLLM(),
        evaluation=ConstantEvaluation(),
        budget=100,
        n_roots=1,
        context_limit=32768,
        checkpoint_dir=tmp_path / "source",
    )
    program = source._forest.add_program(code=TEMPLATE, fitness=1.0, order=1)
    root = source._forest.add_root(program_id=program.id, order=1)
    source._initialization_complete = True
    source._s = 0.0
    source._iteration = 8
    source._n_eval = 20
    source._last_progress_order = 0
    source._last_explore_order = 0
    source._landing_anchor_id = root.id
    source._n_landing = 2
    payload = json.loads(json.dumps(dump_state(source)))

    restored = TraceAADV911(
        llm=ScriptedLLM(),
        evaluation=ConstantEvaluation(),
        budget=100,
        n_roots=1,
        context_limit=32768,
        checkpoint_dir=tmp_path / "restored",
    )
    load_state(restored, payload)

    assert restored._iteration == 8
    assert restored._landing_anchor_id == root.id
    assert restored._n_landing == 2
    assert restored.search_configuration()["protocol_id"] == PROTOCOL_ID


def test_v911_interrupted_resume_reproduces_regime_trajectory(tmp_path: Path) -> None:
    common = {
        "evaluation": ConstantEvaluation(),
        "budget": 30,
        "context_limit": 32768,
        "seed": 9,
    }
    uninterrupted = TraceAADV911(llm=ScriptedLLM(), **common)
    uninterrupted.run()

    interrupted = TraceAADV911(
        llm=ScriptedLLM(),
        checkpoint_dir=tmp_path / "interrupted" / "checkpoints",
        **common,
    )
    interrupted._initialize()
    while interrupted._n_eval < 25:
        decision = interrupted._next_decision()
        interrupted._log_choice(decision)
        interrupted._generate(
            interrupted._prompt(decision.anchor_id, decision.intent),
            anchor_id=decision.anchor_id,
            stage="search",
            iteration=interrupted._iteration,
            intent=decision.intent.value,
        )
    stopped_at = interrupted._n_candidates

    resumed = TraceAADV911(
        llm=ScriptedLLM(start=stopped_at),
        checkpoint_dir=tmp_path / "interrupted" / "checkpoints",
        resume_from=tmp_path / "interrupted" / "checkpoints" / "latest.json",
        **common,
    )
    resumed.run()

    uninterrupted_forest = dump_state(uninterrupted)["forest"]
    resumed_forest = dump_state(resumed)["forest"]
    uninterrupted_programs = [
        {key: value for key, value in item.items() if key != "evaluation_seconds"}
        for item in uninterrupted_forest["programs"]
    ]
    resumed_programs = [
        {key: value for key, value in item.items() if key != "evaluation_seconds"}
        for item in resumed_forest["programs"]
    ]
    assert uninterrupted_programs == resumed_programs
    assert uninterrupted_forest["anchors"] == resumed_forest["anchors"]
    assert uninterrupted_forest["attempts"] == resumed_forest["attempts"]
    assert dump_state(uninterrupted)["last_explore_order"] == 9
    assert dump_state(resumed)["last_explore_order"] == 9
    assert resumed._landing_anchor_id is None
    assert resumed._n_eval == uninterrupted._n_eval == 30


def test_v911_runner_builds_and_records_protocol(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_11",
        backend="server3",
        budget=1000,
        run_name="v911",
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")
    assert isinstance(method, TraceAADV911)
    assert spec.n_init == 8
    assert method.search_configuration() == run._v911_method_params(spec)
    assert method.search_configuration()["protocol_id"] == PROTOCOL_ID
    assert method.search_configuration()["checkpoint_schema_version"] == (
        CHECKPOINT_VERSION
    )
    run_dir, _, _ = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, "v911")
    payload = json.loads((run_dir / "run_config.json").read_text())
    assert payload["method"] == "traceaad_v9_11"
    assert payload["method_params"] == run._v911_method_params(spec)
    assert payload["generator_environment"]["logical_model_name"] == "Qwen3.6-27B"
    assert "backend" not in payload and "llm" not in payload
    method._llm.close()


def test_v911_runner_rejects_non_eight_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly eight"):
        run.make_run_spec(
            task="tsp_construct",
            version="v9_11",
            n_init=1,
            experiments_root=tmp_path,
        )

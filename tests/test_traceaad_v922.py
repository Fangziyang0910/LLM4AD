from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.runners.traceaad.launch_v922 import build_plan, command_for
from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_22 import RunArtifacts, TraceAADV922
from llm4ad.method.traceaad_v9_22.prompt import (
    build_idea_prompt,
    build_realization_prompt,
)
from llm4ad.method.traceaad_v9_22.schema import Hypothesis, ProgramNode


class ToyEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program="def f(x):\n    return 0.0\n",
            task_description="Return a high scalar.",
            safe_evaluate=False,
            timeout_seconds=5,
        )

    def evaluate_program(self, program_str: str, callable_func, **kwargs):
        return float(callable_func(1.0))


class ToyLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.prompts: list[str] = []

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        if "Write only one Idea line" in prompt:
            return f"Idea: direction {self.calls}"
        value = self.calls / 100.0
        return (
            f"Idea: realization {self.calls}\nCode:\n```python\n"
            f"def f(x):\n    return {value}\n```"
        )


def _make_method(tmp_path: Path, llm: LLM, *, budget: int = 6, n_roots: int = 2):
    run_dir = tmp_path / "run"
    return TraceAADV922(
        llm=llm,
        evaluation=ToyEvaluation(),
        artifacts=RunArtifacts(run_dir),
        budget=budget,
        n_roots=n_roots,
        seed=0,
        checkpoint_dir=run_dir / "checkpoints",
        task_key="toy",
    )


def test_v922_runs_paired_realizations_and_freezes_context(tmp_path: Path) -> None:
    method = _make_method(tmp_path, ToyLLM())
    method.run()

    run_dir = tmp_path / "run"
    assert method._n_eval == 6
    assert method._n_calls == 6
    rows = list(csv.DictReader((run_dir / "evaluations.csv").open(newline="")))
    assert len(rows) == 6

    decisions = [
        json.loads(line)
        for line in (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    realizations = [item for item in decisions if item["stage"] == "realization"]
    assert len(realizations) == 4
    groups: dict[tuple[int, str, str], list[dict[str, object]]] = {}
    for item in realizations:
        key = (int(item["batch"]), str(item["proposal"]), str(item["idea"]))
        groups.setdefault(key, []).append(item)
    assert groups
    assert all(len(items) == 2 for items in groups.values())
    assert all(items[0]["prompt"] == items[1]["prompt"] for items in groups.values())

    starts = [
        json.loads(line)
        for line in (run_dir / "mechanism_events.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line).get("event") == "batch_start"
    ]
    assert len(starts) == 1
    assert len(starts[0]["proposal_plan"]) == 2
    summary = json.loads((run_dir / "logs" / "summary.json").read_text(encoding="utf-8"))
    assert summary["mechanism"] == "rank_calibrated_dual_baseline_hypothesis_search"
    assert summary["allocation"] == "hypothesis_ucb_plus_action_ucb"


def test_v922_quality_rank_uses_scaffold_layer_only(tmp_path: Path) -> None:
    method = TraceAADV922(
        ToyLLM(), ToyEvaluation(), budget=2, n_roots=1, task_key="toy"
    )
    method._nodes = {
        1: ProgramNode(1, "root-a", 0.0, None, 1, "a", "root", 1),
        2: ProgramNode(2, "root-b", 10.0, None, 1, "b", "root", 2),
        3: ProgramNode(3, "descendant", 100.0, 1, 1, "c", "realization", 3),
    }
    method._hypotheses = {
        1: Hypothesis(1, "a", 1, 1, 1, None, None, 0),
        2: Hypothesis(2, "b", 2, 2, 2, None, None, 0),
    }

    ranks = method._quality_ranks()
    assert method._quality_rank_count == 2
    assert ranks[1] == pytest.approx(0.0)
    assert ranks[2] == pytest.approx(1.0)


def test_v922_dual_baseline_credit_preserves_repair_direction() -> None:
    values = [0.0, 10.0, 20.0]
    working_response = TraceAADV922._rank_delta(15.0, 10.0, values)
    scaffold_response = TraceAADV922._rank_delta(15.0, 20.0, values)
    breakthrough_response = TraceAADV922._rank_delta(21.0, 20.0, values)

    assert working_response > 0.0
    assert scaffold_response < 0.0
    assert breakthrough_response > 0.0


def test_v922_action_ucb_adapts_proposal_plan(tmp_path: Path) -> None:
    method = TraceAADV922(
        ToyLLM(), ToyEvaluation(), budget=2, n_roots=1, seed=0, task_key="toy"
    )
    method._batch_index = 10
    method._action_stats["continue"].update(trials=50.0, working_improvements=0.0)
    method._action_stats["branch"].update(trials=50.0, working_improvements=50.0)

    proposals, snapshot = method._select_proposals(
        Hypothesis(1, "idea", 1, 1, 1, None, None, 0)
    )
    assert snapshot["branch"]["upper"] > snapshot["continue"]["upper"]
    assert proposals == ["branch", "branch"]


def test_v922_branch_prompt_hides_working_implementation() -> None:
    kwargs = {
        "task_description": "Return a high scalar.",
        "base_code": "def f(x):\n    return 1.0",
        "base_fitness": 1.0,
        "working_code": "def f(x):\n    return 9.0",
        "working_fitness": 9.0,
        "entry_idea": None,
        "formation_history": "[Real Formation Path]\nnone",
        "ledger": "Entry idea: new\nRecent realizations:\nnone",
        "proposal": "branch",
    }
    idea_prompt = build_idea_prompt(**kwargs)
    realization_prompt = build_realization_prompt(
        task_description=kwargs["task_description"],
        idea="new direction",
        base_code=kwargs["base_code"],
        base_fitness=kwargs["base_fitness"],
        working_code=kwargs["working_code"],
        working_fitness=kwargs["working_fitness"],
        formation_history=kwargs["formation_history"],
        ledger=kwargs["ledger"],
        proposal="branch",
    )
    assert "[Current Working Implementation]" not in idea_prompt
    assert "[Current Working Implementation]" not in realization_prompt
    assert "return 9.0" not in idea_prompt
    assert "return 9.0" not in realization_prompt


def test_v922_checkpoint_roundtrip_preserves_new_state(tmp_path: Path) -> None:
    first = _make_method(tmp_path, ToyLLM())
    first.run()
    checkpoint = tmp_path / "run" / "checkpoints" / "latest.json"
    resumed = TraceAADV922(
        ToyLLM(),
        ToyEvaluation(),
        budget=6,
        n_roots=2,
        checkpoint_dir=tmp_path / "resumed" / "checkpoints",
        resume_from=checkpoint,
        task_key="toy",
    )

    assert resumed._n_eval == first._n_eval
    assert resumed._n_calls == first._n_calls
    assert resumed._pending is None
    assert resumed._batch_context is None
    assert resumed._action_stats == first._action_stats
    assert resumed.best is not None
    assert resumed.best.fitness == pytest.approx(first.best.fitness)


def test_v922_launcher_builds_explicit_versioned_plan(tmp_path: Path) -> None:
    plan = build_plan(experiments_root=tmp_path, batch="batch_v922", repeats=1)

    assert len(plan) == 5
    assert plan[0].run_dir == tmp_path / "tsp_construct" / "traceaad_v9_22" / "batch_v922_tsp_rep1"
    command = command_for(plan[0], "local")
    assert command[command.index("--version") + 1] == "v9_22"
    assert command[command.index("--n-init") + 1] == "8"

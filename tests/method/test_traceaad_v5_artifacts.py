"""Contract tests for TraceAAD V5 monitor / artifacts / checkpoint split."""

from __future__ import annotations

import json
from pathlib import Path

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v5 import RunArtifacts, TraceAADV5
from llm4ad.method.traceaad_v5.operators import TraceIdeateOp


TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class _ScriptedLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        text = str(prompt)
        if "Generate a complete implementation" in text or "[Requested Modification]" in text:
            return (
                f"Idea: deterministic candidate {self.calls}\n"
                "```python\n"
                "def choose(value: int) -> int:\n"
                f"    return value + {self.calls}\n"
                "```"
            )
        return "1. Adapt one deterministic offset without adding branches."


class _IncreasingEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            use_numba_accelerate=False,
            safe_evaluate=False,
            timeout_seconds=10,
        )
        self.calls = 0

    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        return float(self.calls)


CANDIDATE_REQUIRED = {
    "sample_order",
    "score",
    "operator",
    "program",
    "idea",
    "code_hash",
    "program_loc",
    "status",
}
CANDIDATE_FORBIDDEN = {
    "parent_id",
    "delta_parent",
    "delta_route_best",
    "delta_global_best",
    "outcome",
    "edge_id",
}

EDGE_REQUIRED = {
    "edge_id",
    "parent_id",
    "child_id",
    "sample_order",
    "iteration",
    "seq",
    "operator",
    "action",
    "anchor_role",
    "primary_trajectory_id",
    "trajectory_id",
}
EDGE_FORBIDDEN = {
    "delta_parent",
    "delta_route_best",
    "delta_global_best",
    "outcome",
    "code_change_ratio",
    "program",
    "score",
    "code_hash",
}

DECISION_FORBIDDEN_FIELDS = {
    "reference_similarity",
    "selected_probability",
    "max_probability",
    "top5_probability_mass",
    "effective_candidate_count",
    "selected_adjusted_score",
    "selected_quality",
    "selected_trend",
    "delta_to_previous_best",
    "delta_parent",
}


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _run_short_search(tmp_path: Path, *, max_sample_nums: int = 4):
    return TraceAADV5(
        llm=_ScriptedLLM(),
        evaluation=_IncreasingEvaluation(),
        artifacts=RunArtifacts(run_dir=tmp_path),
        max_sample_nums=max_sample_nums,
        n_init=2,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceIdeateOp,),
        random_seed=11,
        checkpoint_dir=tmp_path / "checkpoints",
    ).run()


def test_artifacts_layout_is_three_way_split_without_legacy_duplicates(
    tmp_path: Path,
) -> None:
    result = _run_short_search(tmp_path)
    assert result.n_samples == 4

    assert (tmp_path / "logs" / "summary.json").is_file()
    assert (tmp_path / "artifacts" / "candidates.jsonl").is_file()
    assert (tmp_path / "artifacts" / "edges.jsonl").is_file()
    assert (tmp_path / "artifacts" / "llm_calls.jsonl").is_file()
    assert (tmp_path / "artifacts" / "decisions.jsonl").is_file()
    assert (tmp_path / "checkpoints" / "latest.json").is_file()

    # Legacy kitchen-sink paths must not reappear.
    assert not (tmp_path / "logs" / "samples").exists()
    assert not (tmp_path / "logs" / "method_events.jsonl").exists()
    assert not (tmp_path / "logs" / "run_summary.json").exists()
    assert not (tmp_path / "logs" / "llm_calls.jsonl").exists()
    assert not (tmp_path / "logs" / "checkpoints").exists()


def test_candidates_and_edges_store_raw_fields_only(tmp_path: Path) -> None:
    _run_short_search(tmp_path)

    candidates = _load_jsonl(tmp_path / "artifacts" / "candidates.jsonl")
    edges = _load_jsonl(tmp_path / "artifacts" / "edges.jsonl")
    assert len(candidates) == 4
    assert edges, "accepted children should produce edges after init"

    for row in candidates:
        assert CANDIDATE_REQUIRED <= set(row)
        assert CANDIDATE_FORBIDDEN.isdisjoint(row)
        assert row["status"] == "ok"
        assert isinstance(row["program"], str) and row["program"]
        assert row["score"] is not None

    for row in edges:
        assert EDGE_REQUIRED <= set(row)
        assert EDGE_FORBIDDEN.isdisjoint(row)
        assert row["parent_id"] != row["child_id"]
        # Every edge sample_order must point at a candidate.
        assert any(c["sample_order"] == row["sample_order"] for c in candidates)


def test_decisions_are_minimal_raw_facts_without_derived_metrics(
    tmp_path: Path,
) -> None:
    _run_short_search(tmp_path)
    decisions = _load_jsonl(tmp_path / "artifacts" / "decisions.jsonl")
    events = {row["event"] for row in decisions}

    assert "trajectory_created" in events
    assert "trajectory_selected" in events
    assert "operator_selected" in events
    assert "best_updated" in events
    assert "operator_batch" not in events
    assert "program_evaluated" not in events
    assert "child_accepted" not in events
    assert "trajectory_selection" not in events  # old name

    for row in decisions:
        assert DECISION_FORBIDDEN_FIELDS.isdisjoint(row)
        if row["event"] == "trajectory_selected":
            assert set(row) >= {"event", "attempt_id", "trajectory_id"}
        if row["event"] == "operator_selected" and row.get("status") != "no_eligible_operator":
            assert set(row) >= {
                "event",
                "attempt_id",
                "operator",
                "trajectory_id",
                "anchor_id",
                "anchor_role",
            }
        if row["event"] == "best_updated":
            assert row.get("reason") in {"strict_fitness", "tie_shorter"}
            assert "delta_to_previous_best" not in row


def test_llm_calls_keep_metadata_and_drop_successful_prompt_response(
    tmp_path: Path,
) -> None:
    _run_short_search(tmp_path)
    calls = _load_jsonl(tmp_path / "artifacts" / "llm_calls.jsonl")
    assert calls

    stages = {call["stage"] for call in calls}
    assert "init" in stages
    assert "action" in stages or "code" in stages

    for call in calls:
        assert "prompt" not in call
        assert {"stage", "status", "prompt_tokens", "response_tokens"} <= set(call)
        if call["status"] == "ok":
            assert "response" not in call
            assert "parsed_actions" not in call
        if call["stage"] == "action" and call["status"] == "ok":
            assert isinstance(call.get("n_actions"), int)
            assert call["n_actions"] >= 1


def test_summary_is_monitor_only(tmp_path: Path) -> None:
    _run_short_search(tmp_path)
    summary = json.loads((tmp_path / "logs" / "summary.json").read_text())

    assert summary["status"] == "finished"
    assert summary["num_samples"] == 4
    assert "lrr" not in summary
    assert "breakthrough_rate" not in summary
    assert "mean_pcd" not in summary


def test_checkpoint_has_search_state_only_and_resumes(tmp_path: Path) -> None:
    first = TraceAADV5(
        llm=_ScriptedLLM(),
        evaluation=_IncreasingEvaluation(),
        artifacts=RunArtifacts(run_dir=tmp_path),
        max_sample_nums=3,
        n_init=2,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceIdeateOp,),
        random_seed=7,
        checkpoint_dir=tmp_path / "checkpoints",
    ).run()
    payload = json.loads((tmp_path / "checkpoints" / "latest.json").read_text())

    assert "profiler" not in payload
    assert "graph" in payload and "memory" in payload and "rng_state" in payload
    assert payload["total_samples"] == 3
    assert payload["initialization_complete"] is True

    resumed = TraceAADV5(
        llm=_ScriptedLLM(),
        evaluation=_IncreasingEvaluation(),
        artifacts=RunArtifacts(run_dir=tmp_path),
        max_sample_nums=5,
        n_init=2,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceIdeateOp,),
        random_seed=999,
        resume_from=tmp_path / "checkpoints" / "latest.json",
        checkpoint_dir=tmp_path / "checkpoints",
    ).run()

    assert first.n_samples == 3
    assert resumed.n_samples == 5
    assert resumed.n_total_nodes == first.n_total_nodes + 2
    # Checkpoint resume must not depend on regenerating the same RNG seed.
    assert resumed.best_node is not None


def test_runner_v5_wires_run_dir_artifacts_and_checkpoint_paths(
    tmp_path: Path,
) -> None:
    from experiments.runners.traceaad import run

    spec = run.make_run_spec(
        task="online_bin_packing",
        version="v5",
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "one_run")
    assert isinstance(method._artifacts, RunArtifacts)
    assert method._checkpoint_dir == tmp_path / "one_run" / "checkpoints"
    assert method._artifacts._log_dir == tmp_path / "one_run" / "logs"
    assert method._artifacts._artifacts_dir == tmp_path / "one_run" / "artifacts"
    assert run.checkpoint_source(spec, tmp_path / "one_run") == (
        tmp_path / "one_run" / "checkpoints" / "latest.json"
    )


def test_analysis_loader_reads_new_v5_artifact_layout(tmp_path: Path) -> None:
    from experiments.analysis.analyze_process import (
        load_edge_events,
        load_run_summary,
        load_samples,
        resolve_nodes_from_artifacts,
        load_decision_events,
    )

    _run_short_search(tmp_path, max_sample_nums=4)
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "task": "synthetic",
                "experiment_version": "version5",
                "method": "traceaad_v5",
            }
        ),
        encoding="utf-8",
    )

    samples = load_samples(tmp_path)
    edges = load_edge_events(tmp_path)
    decisions = load_decision_events(tmp_path)
    summary = load_run_summary(tmp_path)
    nodes = resolve_nodes_from_artifacts(samples, edges, decisions)

    assert summary["num_samples"] == 4
    assert len(samples) == 4
    assert edges
    assert any(d["event"] == "trajectory_created" for d in decisions)
    assert nodes
    # Init nodes + accepted children should all resolve.
    assert all(node.score is not None for node in nodes.values())
    assert any(node.is_init for node in nodes.values())
    assert any(not node.is_init for node in nodes.values())


def test_eval_failure_lands_in_candidates_not_method_events(tmp_path: Path) -> None:
    class RuntimeFailureLLM(LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            return (
                "Idea: always fail\n"
                "```python\n"
                "def choose(value: int) -> int:\n"
                "    raise ValueError('generated failure')\n"
                "```"
            )

    class ExecutingEvaluation(Evaluation):
        def __init__(self) -> None:
            super().__init__(
                template_program=(
                    "import numpy as np\n\n"
                    "def choose(value: int) -> int:\n"
                    "    return value\n"
                ),
                task_description="Improve choose.",
                use_numba_accelerate=False,
                safe_evaluate=True,
                timeout_seconds=10,
            )

        def evaluate_program(self, program_str, callable_func, **kwargs):
            return float(callable_func(3))

    TraceAADV5(
        llm=RuntimeFailureLLM(),
        evaluation=ExecutingEvaluation(),
        artifacts=RunArtifacts(run_dir=tmp_path),
        max_sample_nums=1,
        n_init=1,
        checkpoint_dir=tmp_path / "checkpoints",
    ).run()

    candidates = _load_jsonl(tmp_path / "artifacts" / "candidates.jsonl")
    assert len(candidates) == 1
    assert candidates[0]["status"] == "runtime_error"
    assert candidates[0]["error_type"] == "ValueError"
    assert candidates[0]["error"] == "generated failure"
    assert candidates[0]["score"] is None
    assert not (tmp_path / "logs" / "method_events.jsonl").exists()
    assert not (tmp_path / "artifacts" / "edges.jsonl").exists() or not _load_jsonl(
        tmp_path / "artifacts" / "edges.jsonl"
    )

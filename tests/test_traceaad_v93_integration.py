from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.method.traceaad_v9_3 import (
    CHECKPOINT_VERSION,
    PROTOCOL_ID,
    TraceAADV93,
)


def test_v93_runner_builds_short_rollout_method(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_3",
        budget=1000,
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")

    assert isinstance(method, TraceAADV93)
    assert spec.method_name == "traceaad_v9_3"
    assert spec.n_init == 8
    assert method.search_configuration()["protocol_id"] == PROTOCOL_ID
    assert (
        method.search_configuration()["checkpoint_schema_version"] == CHECKPOINT_VERSION
    )
    assert method.search_configuration()["rollout_length"] == 3
    assert method.search_configuration()["generation_protocol"] == (
        "trajectory_decision_then_code"
    )


def test_v93_run_config_records_complete_frozen_protocol(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="online_bin_packing",
        version="v9_3",
        budget=1000,
        repeat=2,
        run_name="v9_3_obp_rep2",
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert not resumed
    assert payload["method"] == "traceaad_v9_3"
    assert payload["method_params"] == {
        "protocol_id": PROTOCOL_ID,
        "checkpoint_schema_version": CHECKPOINT_VERSION,
        "max_sample_nums": 1000,
        "initialization_protocol": "strategy_short_rollout_curation",
        "initial_route_pool_size": 8,
        "initial_anchor_count": 6,
        "initial_route_length": 4,
        "initial_route_selection": "best_rollout_representative_by_route_value",
        "generation_protocol": "trajectory_decision_then_code",
        "generation_operator": "trajectory_rollout_step",
        "trajectory_decision_operator": "trajectory_decision",
        "rollout_length": 3,
        "code_representation": "comment_and_docstring_free_ast_canonical",
        "window_protocol": "canonical_formation4_downstream4_depth3",
        "window_size": 8,
        "formation_quota": 4,
        "downstream_quota": 4,
        "downstream_depth": 3,
        "quality_pool_size": 10,
        "quality_policy": "anchor_initialized_mean_rollout_best_absolute_quality",
        "budget_policy": "top10_unverified_first_then_highest_q",
        "invalid_outcome": "rollout_start_anchor_directed_fitness",
        "credit_scope": "selected_rollout_start_anchor_only",
        "eligible_policy": "best_program_per_completed_rollout",
        "ancestor_backup": False,
        "maximize": True,
        "code_max_tokens": 8192,
        "context_token_limit": 24576,
        "max_consecutive_sample_failures": 20,
        "checkpoint_interval": 10,
    }


def test_v93_resume_accepts_only_matching_protocol(tmp_path: Path) -> None:
    original = run.make_run_spec(
        task="tsp_construct",
        version="v9_3",
        budget=1000,
        run_name="matching_v93",
        experiments_root=tmp_path,
    )
    run_dir, run_name, _ = run.resolve_run_dir(original)
    run.write_run_config(original, run_dir, run_name)
    resumed_spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_3",
        budget=1000,
        resume_from=run_dir,
        experiments_root=tmp_path,
    )

    resolved, _, resumed = run.resolve_run_dir(resumed_spec)
    assert resumed
    assert resolved == run_dir


def test_v93_official_runner_fixes_initial_route_pool_to_eight(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly eight"):
        run.make_run_spec(
            task="tsp_construct",
            version="v9_3",
            n_init=10,
            experiments_root=tmp_path,
        )

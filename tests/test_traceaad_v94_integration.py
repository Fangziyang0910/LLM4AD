from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.method.traceaad_v9_4 import (
    CHECKPOINT_VERSION,
    PROTOCOL_ID,
    TraceAADV94,
)


def test_v94_runner_builds_single_step_joint_generation_method(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_4",
        budget=1000,
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")

    assert isinstance(method, TraceAADV94)
    assert spec.method_name == "traceaad_v9_4"
    assert spec.n_init == 8
    assert method.search_configuration()["protocol_id"] == PROTOCOL_ID
    assert (
        method.search_configuration()["checkpoint_schema_version"] == CHECKPOINT_VERSION
    )
    configuration = method.search_configuration()
    assert "rollout_length" not in configuration
    assert "trajectory_decision_operator" not in configuration
    assert configuration["generation_protocol"] == "single_joint_idea_code"
    assert configuration["decision_budget_unit"] == (
        "one_anchor_one_joint_idea_code_one_evaluation"
    )
    assert configuration == run._v94_method_params(spec)


def test_v94_run_config_records_complete_frozen_protocol(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="online_bin_packing",
        version="v9_4",
        budget=1000,
        repeat=2,
        run_name="v9_4_obp_rep2",
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert not resumed
    assert payload["method"] == "traceaad_v9_4"
    assert payload["method_params"] == {
        "protocol_id": PROTOCOL_ID,
        "checkpoint_schema_version": CHECKPOINT_VERSION,
        "max_sample_nums": 1000,
        "initialization_protocol": "strategy_microtrajectory_curation",
        "initial_route_pool_size": 8,
        "initial_anchor_count": 6,
        "initial_route_length": 2,
        "initial_route_selection": "best_endpoint_by_absolute_quality",
        "decision_budget_unit": "one_anchor_one_joint_idea_code_one_evaluation",
        "generation_protocol": "single_joint_idea_code",
        "generation_operator": "trajectory_step",
        "code_representation": "comment_and_docstring_free_ast_canonical",
        "window_protocol": "canonical_formation4_downstream4_depth3",
        "window_size": 8,
        "formation_quota": 4,
        "downstream_quota": 4,
        "downstream_depth": 3,
        "quality_pool_size": 10,
        "quality_policy": (
            "anchor_quality_plus_mean_distance_decayed_descendant_improvement"
        ),
        "budget_policy": "top10_unverified_first_then_four_exploit_one_coverage",
        "quality_exploration_interval": 5,
        "valid_outcome": "positive_descendant_advantage_with_distance_decay",
        "invalid_outcome": "zero_credit_observation",
        "failure_evidence": (
            "local_exact_feedback_and_run_global_top5_exact_patterns_without_failed_code"
        ),
        "strict_breakthrough_definition": (
            "global_strict_directed_fitness_improvement"
        ),
        "credit_scope": "selected_anchor_and_visible_ancestors",
        "trajectory_credit_discount": 0.5,
        "trajectory_credit_depth": 3,
        "eligible_policy": "every_valid_child",
        "ancestor_backup": True,
        "maximize": True,
        "code_max_tokens": 8192,
        "context_token_limit": 24576,
        "max_consecutive_sample_failures": 20,
        "checkpoint_interval": 10,
    }


def test_v94_resume_accepts_only_matching_protocol(tmp_path: Path) -> None:
    original = run.make_run_spec(
        task="tsp_construct",
        version="v9_4",
        budget=1000,
        run_name="matching_v94",
        experiments_root=tmp_path,
    )
    run_dir, run_name, _ = run.resolve_run_dir(original)
    run.write_run_config(original, run_dir, run_name)
    resumed_spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_4",
        budget=1000,
        resume_from=run_dir,
        experiments_root=tmp_path,
    )

    resolved, _, resumed = run.resolve_run_dir(resumed_spec)
    assert resumed
    assert resolved == run_dir


def test_v94_official_runner_fixes_initial_route_pool_to_eight(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly eight"):
        run.make_run_spec(
            task="tsp_construct",
            version="v9_4",
            n_init=10,
            experiments_root=tmp_path,
        )

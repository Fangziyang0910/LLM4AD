from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.method.traceaad_v9_2 import (
    CHECKPOINT_VERSION,
    PROTOCOL_ID,
    TraceAADV92,
)


def test_v92_runner_builds_independent_anchor_window_method(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_2",
        budget=17,
        seed=4,
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")

    assert isinstance(method, TraceAADV92)
    assert spec.method_name == "traceaad_v9_2"
    assert spec.n_init == 8
    assert method.search_configuration()["protocol_id"] == PROTOCOL_ID
    assert (
        method.search_configuration()["checkpoint_schema_version"] == CHECKPOINT_VERSION
    )
    assert method.search_configuration()["generation_operator"] == "trajectory_step"
    assert not hasattr(method, "_operators")
    assert not hasattr(method, "_rng")


def test_v92_run_config_records_complete_frozen_protocol(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="online_bin_packing",
        version="v9_2",
        budget=17,
        seed=3,
        repeat=2,
        run_name="v9_2_obp_rep2",
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    params = payload["method_params"]

    assert not resumed
    assert payload["method"] == "traceaad_v9_2"
    assert params == {
        "protocol_id": PROTOCOL_ID,
        "checkpoint_schema_version": CHECKPOINT_VERSION,
        "max_sample_nums": 17,
        "initialization_protocol": "strategy_microtrajectory_curation",
        "initial_route_pool_size": 8,
        "initial_anchor_count": 6,
        "initial_route_length": 2,
        "initial_route_selection": "best_endpoint_then_anchor_budget_value",
        "generation_protocol": "single_idea_code",
        "generation_operator": "trajectory_step",
        "code_representation": "comment_and_docstring_free_ast_canonical",
        "window_protocol": "canonical_formation4_downstream4_depth3",
        "window_size": 8,
        "formation_quota": 4,
        "downstream_quota": 4,
        "downstream_depth": 3,
        "quality_pool_size": 10,
        "quality_policy": "raw_directed_anchor_running_mean",
        "budget_policy": "top10_unverified_first_then_highest_q",
        "invalid_outcome": "anchor_directed_fitness",
        "credit_scope": "selected_anchor_only",
        "ancestor_backup": False,
        "maximize": True,
        "code_max_tokens": 8192,
        "context_token_limit": 24576,
        "max_consecutive_sample_failures": 20,
        "checkpoint_interval": 10,
    }


def test_v92_resume_accepts_only_matching_v92_protocol(tmp_path: Path) -> None:
    original = run.make_run_spec(
        task="tsp_construct",
        version="v9_2",
        budget=17,
        run_name="matching_v92",
        experiments_root=tmp_path,
    )
    run_dir, run_name, _ = run.resolve_run_dir(original)
    run.write_run_config(original, run_dir, run_name)
    resumed_spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_2",
        budget=17,
        resume_from=run_dir,
        experiments_root=tmp_path,
    )

    resolved, _, resumed = run.resolve_run_dir(resumed_spec)
    assert resumed
    assert resolved == run_dir


def test_v92_official_runner_fixes_initial_route_pool_to_eight(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="exactly eight"):
        run.make_run_spec(
            task="tsp_construct",
            version="v9_2",
            n_init=10,
            experiments_root=tmp_path,
        )

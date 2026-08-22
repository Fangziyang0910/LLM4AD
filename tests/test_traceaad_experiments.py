from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.method.traceaad_v4 import TraceAADV4
from llm4ad.method.traceaad_v5 import TraceAADV5
from llm4ad.method.traceaad_v8 import TraceAADV8
from llm4ad.method.traceaad_v9 import TraceAADV9
from llm4ad.method.traceaad_v9_7 import TraceAADV97
from llm4ad.method.traceaad_v9_14 import TraceAADV914
from llm4ad.method.traceaad_v9_15 import TraceAADV915
from llm4ad.method.traceaad_v9_16 import TraceAADV916


@pytest.mark.parametrize("task", run.TASKS)
@pytest.mark.parametrize("version", run.VERSIONS)
def test_unified_runner_builds_each_task_and_version(
    tmp_path: Path,
    task: run.TaskName,
    version: run.VersionName,
) -> None:
    spec = run.make_run_spec(
        task=task,
        version=version,
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")

    expected_type = {
        "v4": TraceAADV4,
        "v5": TraceAADV5,
        "v8": TraceAADV8,
        "v9": TraceAADV9,
        "v9_7": TraceAADV97,
        "v9_14": TraceAADV914,
        "v9_15": TraceAADV915,
        "v9_16": TraceAADV916,
    }[version]
    assert isinstance(method, expected_type)
    assert spec.experiment_root == tmp_path / task / f"traceaad_{version}"

    if version == "v4":
        assert method._llm.max_tokens == 16384
    else:
        assert method._llm.max_tokens == 8192
        if version == "v9_15":
            assert method._error_handling is True
            assert method._error_retries == 2
        if version in {"v8", "v9"}:
            assert spec.n_init == 10
            assert not hasattr(method, "_action_max_tokens")
            assert method._offspring_per_iteration == 2
            assert method._context_token_limit == 24576
            assert not hasattr(method, "_dual_probability")
        elif version == "v5":
            assert spec.n_init == 30
            assert method._action_max_tokens == 1024
        else:
            assert spec.n_init == 8
            assert method._n_roots == 8
            assert not hasattr(method, "_context_limit")
            assert not hasattr(method, "_operators")
        assert not hasattr(method, "_global_experience")


def test_runner_writes_one_reproducible_config_per_run(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="online_bin_packing",
        version="v5",
        backend="local",
        budget=17,
        seed=3,
        repeat=3,
        run_name="batch_obp_v5_rep3",
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert not resumed
    assert payload["task"] == "online_bin_packing"
    assert payload["method"] == "traceaad_v5"
    assert payload["repeat"] == 3
    assert payload["backend"] == "local"
    assert payload["method_params"]["max_sample_nums"] == 17
    assert payload["method_params"]["random_seed"] == 3
    assert "api_key" not in payload["llm"]
    assert payload["llm"]["api_key_configured"] is False


def test_v915_config_records_retry_policy(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_15",
        repeat=1,
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert not resumed
    assert payload["method"] == "traceaad_v9_15"
    assert payload["method_params"]["error_handling"] is True
    assert payload["method_params"]["error_retries"] == 2
    assert payload["method_params"]["retry_policy"] == "two_bounded_repairs"
    assert payload["method_params"]["retry_budget"] == "initial_candidates"


def test_v8_runner_records_tree_protocol_without_population_controls(
    tmp_path: Path,
) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v8",
        experiments_root=tmp_path,
    )
    run_dir, run_name, _ = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    params = payload["method_params"]
    assert params["n_init"] == 10
    assert params["exploration_constant"] == 0.1
    assert params["expansion_prior_weight"] == 1.0
    assert params["offspring_per_iteration"] == 2
    assert params["generation_protocol"] == "direct_code"
    assert params["quality_normalization"] == "global_midrank_percentile"
    assert params["expansion_policy"] == "adaptive_new_child_uct"
    assert params["expansion_reward"] == "batch_subtree_best_midrank"
    assert params["failed_expansion_reward"] == 0.0
    assert params["root_expansion"] is False
    assert params["direct_child_top_count"] == 4
    assert params["operators"] == [
        "trace_ideate",
        "trace_refine",
        "trace_synthesize",
        "trace_transfer",
    ]
    assert "max_active_trajectories" not in params
    assert "value_weights" not in params
    assert "elite_count" not in params
    assert "actions_per_iteration" not in params
    assert "action_max_tokens" not in params


def test_resume_uses_version_specific_checkpoint_source(tmp_path: Path) -> None:
    for version in run.VERSIONS:
        run_dir = tmp_path / version
        run_dir.mkdir()
        original_spec = run.make_run_spec(
            task="tsp_construct",
            version=version,
            experiments_root=tmp_path,
        )
        run.write_run_config(original_spec, run_dir, run_dir.name)
        spec = run.make_run_spec(
            task="tsp_construct",
            version=version,
            resume_from=run_dir,
            experiments_root=tmp_path,
        )
        resolved, _, resumed = run.resolve_run_dir(spec)

        assert resumed
        assert resolved == run_dir
        expected = run_dir / "checkpoints" / "latest.json"
        assert run.checkpoint_source(spec, resolved) == expected


def test_v8_resume_rejects_changed_experiment_configuration(tmp_path: Path) -> None:
    run_dir = tmp_path / "v8"
    run_dir.mkdir()
    original = run.make_run_spec(
        task="tsp_construct",
        version="v8",
        budget=100,
        seed=3,
        experiments_root=tmp_path,
    )
    run.write_run_config(original, run_dir, run_dir.name)
    changed = run.make_run_spec(
        task="tsp_construct",
        version="v8",
        budget=101,
        seed=3,
        resume_from=run_dir,
        experiments_root=tmp_path,
    )
    with pytest.raises(ValueError, match="resume config mismatch"):
        run.resolve_run_dir(changed)


def test_aco_tasks_default_to_four_local_eval_workers() -> None:
    from experiments.runners._common import DEFAULT_ACO_EVAL_WORKERS, build_task

    assert DEFAULT_ACO_EVAL_WORKERS == 4
    cvrp, cvrp_kwargs = build_task("cvrp_aco", None)
    op, op_kwargs = build_task("op_aco", None)
    assert cvrp.n_workers == 4
    assert op.n_workers == 4
    assert cvrp_kwargs["n_workers"] == 4
    assert op_kwargs["n_workers"] == 4


def test_old_task_specific_traceaad_runners_are_removed() -> None:
    for task in run.TASKS:
        for version in run.VERSIONS:
            assert not Path(
                f"experiments/{task}/traceaad_{version}/run_experiment.py"
            ).exists()

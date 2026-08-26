from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.method.traceaad_v9_7 import TraceAADV97
from llm4ad.method.traceaad_v9_14 import TraceAADV914
from llm4ad.method.traceaad_v9_16 import TraceAADV916
from llm4ad.method.traceaad_v9_17 import TraceAADV917
from llm4ad.method.traceaad_v9_18 import TraceAADV918


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
        "v9_7": TraceAADV97,
        "v9_14": TraceAADV914,
        "v9_16": TraceAADV916,
        "v9_17": TraceAADV917,
        "v9_17_fixed_cycle": TraceAADV917,
        "v9_18_q_atomic": TraceAADV918,
        "v9_18_q_opportunity": TraceAADV918,
        "v9_18_facts": TraceAADV918,
    }[version]
    assert isinstance(method, expected_type)
    assert spec.experiment_root == tmp_path / task / f"traceaad_{version}"

    assert method._llm.max_tokens == 8192
    assert spec.n_init == 8
    assert method._n_roots == 8
    if version in {"v9_17", "v9_17_fixed_cycle"}:
        assert method._active_capacity == 8
        assert method._adaptive_sweeps is (version == "v9_17")
    if version in {"v9_18_q_atomic", "v9_18_q_opportunity", "v9_18_facts"}:
        assert method._allocation_mode == (
            "opportunity" if version == "v9_18_q_opportunity" else "q"
        )
        assert method._explore_context == (
            "facts" if version == "v9_18_facts" else "legacy"
        )
    assert not hasattr(method, "_context_limit")
    assert not hasattr(method, "_operators")
    assert not hasattr(method, "_global_experience")


def test_runner_writes_one_reproducible_config_per_run(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="online_bin_packing",
        version="v9_7",
        backend="local",
        budget=17,
        seed=3,
        repeat=3,
        run_name="batch_obp_v9_7_rep3",
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert not resumed
    assert payload["task"] == "online_bin_packing"
    assert payload["method"] == "traceaad_v9_7"
    assert payload["repeat"] == 3
    assert payload["method_params"]["budget"] == 17
    assert payload["method_params"]["seed"] == 3
    assert payload["generator_environment"]["sampling_seed"] == 3


def test_traceaad_config_records_llm_metadata(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_18_q_atomic",
        backend="server3",
        repeat=2,
        seed=7,
        run_name="v9_18_metadata_rep2",
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert not resumed
    assert payload["run_name"] == "v9_18_metadata_rep2"
    assert payload["backend"] == "server3"
    assert payload["seed"] == 7
    assert payload["llm"]["model"] == spec.model
    assert payload["llm"]["max_tokens"] == 8192


def test_v917_config_records_the_fixed_hypothesis_protocol(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_17",
        repeat=1,
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert not resumed
    params = payload["method_params"]
    assert params["n_roots"] == 8
    assert params["active_capacity"] == 8
    assert params["block_horizon"] == 3
    assert params["max_history"] == 8
    assert params["competition_rank"] == "frontier_quality_then_creation"
    assert params["development_continuation"] == "positive_block_gain"
    assert params["retry_budget"] == "primary_candidates"
    assert "refine_probability" not in params
    assert "explore_probability" not in params


def test_v917_fixed_cycle_config_differs_only_in_scheduler_rule() -> None:
    adaptive = run.make_run_spec(task="tsp_construct", version="v9_17")
    fixed = run.make_run_spec(task="tsp_construct", version="v9_17_fixed_cycle")
    adaptive_params = run._v917_method_params(adaptive)
    fixed_params = run._v917_method_params(fixed)
    differing = {
        key
        for key in adaptive_params | fixed_params
        if adaptive_params.get(key) != fixed_params.get(key)
    }
    assert differing == {"development_continuation"}
    assert fixed_params["development_continuation"] == "fixed_cycle_after_full_sweep"


def test_v918_initialization_checkpoint_is_allowed(tmp_path: Path) -> None:
    source = run.make_run_spec(
        task="tsp_construct",
        version="v9_18_q_atomic",
        budget=8,
        experiments_root=tmp_path,
    )
    checkpoint = tmp_path / "initialization" / "latest.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text("{}", encoding="utf-8")
    fork = run.make_run_spec(
        task="tsp_construct",
        version="v9_18_q_opportunity",
        budget=1000,
        initialization_checkpoint=checkpoint,
        experiments_root=tmp_path,
    )
    assert fork.initialization_checkpoint == checkpoint.resolve()
    assert source.n_init == fork.n_init == 8


def test_seed_paired_artifacts_accepts_checkpoint_under_checkpoints_dir(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "checkpoints").mkdir(parents=True)
    (source / "checkpoints" / "latest.json").write_text("{}", encoding="utf-8")
    (source / "evaluations.csv").write_text("header\n", encoding="utf-8")
    (source / "mechanism_events.jsonl").write_text("{}\n", encoding="utf-8")
    target = tmp_path / "target"
    run._seed_paired_artifacts(source / "checkpoints" / "latest.json", target)
    assert (target / "evaluations.csv").read_text(encoding="utf-8") == "header\n"
    assert (target / "mechanism_events.jsonl").is_file()


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


def test_resume_rejects_changed_experiment_configuration(tmp_path: Path) -> None:
    run_dir = tmp_path / "v9_7"
    run_dir.mkdir()
    original = run.make_run_spec(
        task="tsp_construct",
        version="v9_7",
        budget=100,
        seed=3,
        experiments_root=tmp_path,
    )
    run.write_run_config(original, run_dir, run_dir.name)
    changed = run.make_run_spec(
        task="tsp_construct",
        version="v9_7",
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

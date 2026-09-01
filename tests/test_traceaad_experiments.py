from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from experiments.runners._common import ALL_TASKS
from llm4ad.method.traceaad_v9_7 import TraceAADV97
from llm4ad.method.traceaad_v9_14 import TraceAADV914
from llm4ad.method.traceaad_v9_16 import TraceAADV916
from llm4ad.method.traceaad_v9_17 import TraceAADV917
from llm4ad.method.traceaad_v9_18 import TraceAADV918
from llm4ad.method.traceaad_v9_19 import TraceAADV919
from llm4ad.method.traceaad_v9_20 import TraceAADV920
from llm4ad.method.traceaad_v9_21 import TraceAADV921
from llm4ad.method.traceaad_v10 import TraceAADV10

RUN_MODULES = {
    version: importlib.import_module(f"experiments.runners.traceaad_{version}.run")
    for version in (
        "v9_7",
        "v9_14",
        "v9_16",
        "v9_17",
        "v9_17_fixed_cycle",
        "v9_18_q_atomic",
        "v9_18_q_opportunity",
        "v9_19",
        "v9_20",
        "v9_21",
        "v10",
    )
}
VERSIONS = tuple(RUN_MODULES)
TASKS = tuple(ALL_TASKS)

EXPECTED_METHOD_TYPE = {
    "v9_7": TraceAADV97,
    "v9_14": TraceAADV914,
    "v9_16": TraceAADV916,
    "v9_17": TraceAADV917,
    "v9_17_fixed_cycle": TraceAADV917,
    "v9_18_q_atomic": TraceAADV918,
    "v9_18_q_opportunity": TraceAADV918,
    "v9_19": TraceAADV919,
    "v9_20": TraceAADV920,
    "v9_21": TraceAADV921,
    "v10": TraceAADV10,
}


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("version", VERSIONS)
def test_each_run_module_builds_its_task(tmp_path: Path, task: str, version: str) -> None:
    runner = RUN_MODULES[version]
    spec = runner.make_run_spec(task=task, experiments_root=tmp_path)
    method = runner.build_method(spec, tmp_path / "run")

    assert isinstance(method, EXPECTED_METHOD_TYPE[version])
    assert spec.experiment_root == tmp_path / task / f"traceaad_{version}"

    assert method._llm.max_tokens == 8192
    assert spec.n_init == 8
    assert method._n_roots == 8
    if version in {"v9_17", "v9_17_fixed_cycle"}:
        assert method._active_capacity == 8
        assert method._adaptive_sweeps is (version == "v9_17")
    if version in {"v9_18_q_atomic", "v9_18_q_opportunity"}:
        assert method._allocation_mode == (
            "opportunity" if version == "v9_18_q_opportunity" else "q"
        )
        assert method._explore_context == "legacy"
    assert not hasattr(method, "_context_limit")
    assert not hasattr(method, "_operators")
    assert not hasattr(method, "_global_experience")


def test_runner_writes_one_reproducible_config_per_run(tmp_path: Path) -> None:
    runner = RUN_MODULES["v9_7"]
    spec = runner.make_run_spec(
        task="online_bin_packing",
        backend="local",
        budget=17,
        seed=3,
        repeat=3,
        run_name="batch_obp_v9_7_rep3",
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = runner.resolve_run_dir(spec)
    runner.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert not resumed
    assert payload["task"] == "online_bin_packing"
    assert payload["method"] == "traceaad_v9_7"
    assert payload["repeat"] == 3
    assert payload["method_params"]["budget"] == 17
    assert payload["method_params"]["seed"] == 3
    assert payload["generator_environment"]["sampling_seed"] == 3


def test_traceaad_config_records_llm_metadata(tmp_path: Path) -> None:
    runner = RUN_MODULES["v9_18_q_atomic"]
    spec = runner.make_run_spec(
        task="tsp_construct",
        backend="server3",
        repeat=2,
        seed=7,
        run_name="v9_18_metadata_rep2",
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = runner.resolve_run_dir(spec)
    runner.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert not resumed
    assert payload["run_name"] == "v9_18_metadata_rep2"
    assert payload["backend"] == "server3"
    assert payload["seed"] == 7
    assert payload["llm"]["model"] == spec.model
    assert payload["llm"]["max_tokens"] == 8192


def test_v917_config_records_the_fixed_hypothesis_protocol(tmp_path: Path) -> None:
    runner = RUN_MODULES["v9_17"]
    spec = runner.make_run_spec(
        task="tsp_construct",
        repeat=1,
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = runner.resolve_run_dir(spec)
    runner.write_run_config(spec, run_dir, run_name)
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
    adaptive = RUN_MODULES["v9_17"].make_run_spec(task="tsp_construct")
    fixed = RUN_MODULES["v9_17_fixed_cycle"].make_run_spec(task="tsp_construct")
    adaptive_params = RUN_MODULES["v9_17"]._method_params(adaptive)
    fixed_params = RUN_MODULES["v9_17_fixed_cycle"]._method_params(fixed)
    differing = {
        key
        for key in adaptive_params | fixed_params
        if adaptive_params.get(key) != fixed_params.get(key)
    }
    assert differing == {"development_continuation"}
    assert fixed_params["development_continuation"] == "fixed_cycle_after_full_sweep"


def test_v918_initialization_checkpoint_is_allowed(tmp_path: Path) -> None:
    source = RUN_MODULES["v9_18_q_atomic"].make_run_spec(
        task="tsp_construct",
        budget=8,
        experiments_root=tmp_path,
    )
    checkpoint = tmp_path / "initialization" / "latest.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text("{}", encoding="utf-8")
    fork = RUN_MODULES["v9_18_q_opportunity"].make_run_spec(
        task="tsp_construct",
        budget=1000,
        initialization_checkpoint=checkpoint,
        experiments_root=tmp_path,
    )
    assert fork.initialization_checkpoint == checkpoint.resolve()
    assert source.n_init == fork.n_init == 8


def test_seed_paired_artifacts_accepts_checkpoint_under_checkpoints_dir(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "checkpoints").mkdir(parents=True)
    (source / "checkpoints" / "latest.json").write_text("{}", encoding="utf-8")
    (source / "evaluations.csv").write_text("header\n", encoding="utf-8")
    (source / "mechanism_events.jsonl").write_text("{}\n", encoding="utf-8")
    target = tmp_path / "target"
    RUN_MODULES["v9_18_q_atomic"]._seed_paired_artifacts(
        source / "checkpoints" / "latest.json", target
    )
    assert (target / "evaluations.csv").read_text(encoding="utf-8") == "header\n"
    assert (target / "mechanism_events.jsonl").is_file()


def test_resume_uses_version_specific_checkpoint_source(tmp_path: Path) -> None:
    for version, runner in RUN_MODULES.items():
        run_dir = tmp_path / version
        run_dir.mkdir()
        original_spec = runner.make_run_spec(
            task="tsp_construct",
            experiments_root=tmp_path,
        )
        runner.write_run_config(original_spec, run_dir, run_dir.name)
        spec = runner.make_run_spec(
            task="tsp_construct",
            resume_from=run_dir,
            experiments_root=tmp_path,
        )
        resolved, _, resumed = runner.resolve_run_dir(spec)

        assert resumed
        assert resolved == run_dir
        expected = run_dir / "checkpoints" / "latest.json"
        assert runner.checkpoint_source(spec, resolved) == expected


def test_resume_rejects_changed_experiment_configuration(tmp_path: Path) -> None:
    runner = RUN_MODULES["v9_7"]
    run_dir = tmp_path / "v9_7"
    run_dir.mkdir()
    original = runner.make_run_spec(
        task="tsp_construct",
        budget=100,
        seed=3,
        experiments_root=tmp_path,
    )
    runner.write_run_config(original, run_dir, run_dir.name)
    changed = runner.make_run_spec(
        task="tsp_construct",
        budget=101,
        seed=3,
        resume_from=run_dir,
        experiments_root=tmp_path,
    )
    with pytest.raises(ValueError, match="resume config mismatch"):
        runner.resolve_run_dir(changed)


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
    for task in TASKS:
        for version in VERSIONS:
            assert not Path(
                f"experiments/{task}/traceaad_{version}/run_experiment.py"
            ).exists()
    assert not Path("experiments/runners/traceaad/run.py").exists()

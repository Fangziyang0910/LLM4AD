from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from experiments.runners._common import ALL_TASKS
from llm4ad.method.traceaad_v9_7 import TraceAADV97
from llm4ad.method.traceaad_v9_14 import TraceAADV914
from llm4ad.method.traceaad_v9_16 import TraceAADV916

RUN_MODULES = {
    version: importlib.import_module(f"experiments.runners.traceaad_{version}.run")
    for version in (
        "v9_7",
        "v9_14",
        "v9_16",
    )
}
VERSIONS = tuple(RUN_MODULES)
TASKS = tuple(ALL_TASKS)

EXPECTED_METHOD_TYPE = {
    "v9_7": TraceAADV97,
    "v9_14": TraceAADV914,
    "v9_16": TraceAADV916,
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
    runner = RUN_MODULES["v9_16"]
    spec = runner.make_run_spec(
        task="tsp_construct",
        backend="server3",
        repeat=2,
        seed=7,
        run_name="v9_16_metadata_rep2",
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = runner.resolve_run_dir(spec)
    runner.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert not resumed
    assert payload["run_name"] == "v9_16_metadata_rep2"
    assert payload["backend"] == "server3"
    assert payload["seed"] == 7
    assert payload["llm"]["model"] == spec.model
    assert payload["llm"]["max_tokens"] == 8192


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

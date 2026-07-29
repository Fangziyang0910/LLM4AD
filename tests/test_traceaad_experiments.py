from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.traceaad import launch, run
from llm4ad.method.traceaad_v4 import TraceAADV4
from llm4ad.method.traceaad_v5 import TraceAADV5


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
    method = run.build_method(spec, tmp_path / "logs")

    expected_type = TraceAADV4 if version == "v4" else TraceAADV5
    assert isinstance(method, expected_type)
    assert spec.experiment_root == tmp_path / task / f"traceaad_{version}"
    assert spec.experiment_version == f"version{version.removeprefix('v')}"

    if version == "v4":
        assert method._llm.max_tokens == 16384
    else:
        assert method._llm.max_tokens == 8192
        assert method._output_token_reserve == 8192
        assert method._action_max_tokens == 1024
        assert method._max_context_tokens is None
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
    assert payload["experiment_version"] == "version5"
    assert payload["repeat"] == 3
    assert payload["backend"] == "local"
    assert payload["method_params"]["max_sample_nums"] == 17
    assert payload["method_params"]["random_seed"] == 3
    assert "api_key" not in payload["llm"]
    assert payload["llm"]["api_key_configured"] is False


def test_resume_uses_version_specific_checkpoint_source(tmp_path: Path) -> None:
    for version in run.VERSIONS:
        method = f"traceaad_{version}"
        run_dir = tmp_path / version
        run_dir.mkdir()
        (run_dir / "run_config.json").write_text(
            json.dumps({"task": "tsp_construct", "method": method}),
            encoding="utf-8",
        )
        spec = run.make_run_spec(
            task="tsp_construct",
            version=version,
            resume_from=run_dir,
            experiments_root=tmp_path,
        )
        resolved, _, resumed = run.resolve_run_dir(spec)

        assert resumed
        assert resolved == run_dir
        expected = (
            run_dir
            if version == "v4"
            else run_dir / "logs" / "checkpoints" / "latest.json"
        )
        assert run.checkpoint_source(spec, resolved) == expected


def test_batch_launcher_builds_independent_repeat_commands() -> None:
    args = launch.build_parser().parse_args(
        [
            "--task",
            "online_bin_packing",
            "--version",
            "v4",
            "--backend",
            "local",
            "--repeats",
            "3",
            "--batch",
            "20260729_230434",
            "--dry-run",
        ]
    )
    plan = launch.build_launch_plan(args)

    assert [item.repeat for item in plan] == [1, 2, 3]
    assert len({item.session for item in plan}) == 3
    assert len({item.run_dir for item in plan}) == 3
    assert plan[0].run_name == "20260729_230434_obp_v4_rep1"
    assert "--repeat" in plan[0].command
    assert "--run-name" in plan[0].command


def test_old_task_specific_traceaad_runners_are_removed() -> None:
    for task in run.TASKS:
        for version in run.VERSIONS:
            assert not Path(
                f"experiments/{task}/traceaad_{version}/run_experiment.py"
            ).exists()

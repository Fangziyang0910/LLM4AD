from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import launch, run, schedule
from llm4ad.method.traceaad_v4 import TraceAADV4
from llm4ad.method.traceaad_v5 import TraceAADV5
from llm4ad.method.traceaad_v8 import PROTOCOL_ID as V8_PROTOCOL_ID
from llm4ad.method.traceaad_v8 import TraceAADV8
from llm4ad.method.traceaad_v9 import TraceAADV9


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
    }[version]
    assert isinstance(method, expected_type)
    assert spec.experiment_root == tmp_path / task / f"traceaad_{version}"
    assert spec.experiment_version == f"version{version.removeprefix('v')}"

    if version == "v4":
        assert method._llm.max_tokens == 16384
    else:
        assert method._llm.max_tokens == 8192
        if version in {"v8", "v9"}:
            assert spec.n_init == 10
            assert not hasattr(method, "_action_max_tokens")
            assert method._offspring_per_iteration == 2
            assert method._context_token_limit == 24576
            assert not hasattr(method, "_dual_probability")
        else:
            assert spec.n_init == 30
            assert method._action_max_tokens == 1024
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
    assert payload["experiment_version"] == "version8"
    assert params["protocol_id"] == V8_PROTOCOL_ID
    assert params["checkpoint_schema_version"] == 3
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


def test_batch_launcher_uses_version_specific_initialization_defaults() -> None:
    v8_args = launch.build_parser().parse_args(
        ["--task", "tsp_construct", "--version", "v8", "--dry-run"]
    )
    v5_args = launch.build_parser().parse_args(
        ["--task", "tsp_construct", "--version", "v5", "--dry-run"]
    )

    v8_command = launch.build_launch_plan(v8_args)[0].command
    v5_command = launch.build_launch_plan(v5_args)[0].command
    assert v8_command[v8_command.index("--n-init") + 1] == "10"
    assert v5_command[v5_command.index("--n-init") + 1] == "30"


def test_v82_scheduler_builds_four_tasks_by_three_repeats(tmp_path: Path) -> None:
    state = schedule.build_state(
        batch="20260804_220000",
        budget=1000,
        n_init=10,
        context_token_limit=24576,
        experiments_root=tmp_path,
    )
    jobs = state["jobs"]

    assert len(jobs) == 12
    assert [(job["task"], job["repeat"]) for job in jobs[:4]] == [
        (task, 1) for task in run.TASKS
    ]
    assert {job["seed"] for job in jobs} == {1, 2, 3}
    assert len({job["session"] for job in jobs}) == 12
    assert all("v82_20260804_220000" in job["run_name"] for job in jobs)


def test_v82_scheduler_counts_inherited_worker_command_once() -> None:
    zhong = "python -m experiments.runners.traceaad.run --task op_aco --backend zhong"
    server1 = (
        "python -m experiments.runners.reevo.run --task tsp_construct --backend server1"
    )
    rows = [
        (100, 1, zhong),
        (101, 100, zhong),
        (102, 101, zhong),
        (200, 1, server1),
        (300, 1, "python -c from multiprocessing.spawn import spawn_main"),
    ]

    assert schedule.count_backend_usage(rows) == {
        "zhong": 1,
        "server1": 1,
        "local": 0,
    }


def test_v82_scheduler_assigns_only_available_slots(tmp_path: Path) -> None:
    state = schedule.build_state(
        batch="assignment",
        budget=1000,
        n_init=10,
        context_token_limit=24576,
        experiments_root=tmp_path,
    )
    assigned = schedule.assign_pending(
        state,
        {"zhong": 2, "server1": 2, "local": 1},
    )

    assert [job["backend"] for job in assigned] == [
        "zhong",
        "zhong",
        "server1",
        "server1",
        "local",
    ]
    assert len(assigned) == 5


def test_v82_scheduler_state_reload_prevents_duplicate_assignment(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "scheduler.json"
    state = schedule.build_state(
        batch="resume",
        budget=1000,
        n_init=10,
        context_token_limit=24576,
        experiments_root=tmp_path,
    )
    assigned = schedule.assign_pending(
        state,
        {"zhong": 1, "server1": 0, "local": 0},
    )
    assigned[0]["status"] = "running"
    schedule.save_state(state, state_path)

    restored = schedule.load_state(state_path)
    next_jobs = schedule.assign_pending(
        restored,
        {"zhong": 1, "server1": 0, "local": 0},
    )

    assert next_jobs[0]["run_name"] != assigned[0]["run_name"]


def test_v82_scheduler_reads_terminal_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = schedule.build_state(
        batch="finished",
        budget=1000,
        n_init=10,
        context_token_limit=24576,
        experiments_root=tmp_path,
    )
    job = state["jobs"][0]
    summary = Path(job["run_dir"]) / "logs" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"status": "finished"}\n', encoding="utf-8")
    monkeypatch.setattr(schedule, "_session_exists", lambda _session: False)

    assert schedule.reconcile_job(job)
    assert job["status"] == "finished"


def test_scheduler_rejects_cross_version_state_reload(tmp_path: Path) -> None:
    state = schedule.build_state(
        batch="protocol_guard",
        budget=1000,
        n_init=10,
        context_token_limit=24576,
        version="v9",
        experiments_root=tmp_path,
    )
    path = tmp_path / "scheduler.json"
    schedule.save_state(state, path)

    with pytest.raises(ValueError, match="does not match requested version"):
        schedule.load_state(path, expected_version="v8")

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["jobs"][0]["version"] = "v8"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="job version"):
        schedule.load_state(path)


def test_scheduler_reconciles_stalled_terminal_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = schedule.build_state(
        batch="stalled",
        budget=1000,
        n_init=10,
        context_token_limit=24576,
        experiments_root=tmp_path,
    )
    job = state["jobs"][0]
    summary = Path(job["run_dir"]) / "logs" / "summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text('{"status": "stalled"}\n', encoding="utf-8")
    monkeypatch.setattr(schedule, "_session_exists", lambda _session: True)

    assert schedule.reconcile_job(job)
    assert job["status"] == "stalled"


def test_old_task_specific_traceaad_runners_are_removed() -> None:
    for task in run.TASKS:
        for version in run.VERSIONS:
            assert not Path(
                f"experiments/{task}/traceaad_{version}/run_experiment.py"
            ).exists()

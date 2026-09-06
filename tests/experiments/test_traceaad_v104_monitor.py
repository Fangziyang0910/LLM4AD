import pytest

from experiments.traceaad_v10_4.monitor import (
    TASKS_METADATA,
    MonitorDataEngine,
)


def test_v104_monitor_task_metadata_consistency() -> None:
    expected_tasks = {
        "tsp_construct",
        "cvrp_aco",
        "op_aco",
        "online_bin_packing",
        "vrptw_construct",
    }
    actual_tasks = {t["key"] for t in TASKS_METADATA}
    assert actual_tasks == expected_tasks


def test_v104_monitor_engine_overview_and_runs() -> None:
    engine = MonitorDataEngine()
    overview = engine.get_overview()

    assert overview["version"] == "V10.4"
    assert overview["version_id"] == "v10_4"
    global_sum = overview["global_summary"]
    assert global_sum["total_runs"] == 15
    assert global_sum["total_budget"] == 15000
    assert global_sum["total_evals"] >= 0
    assert len(overview["tasks"]) == 5

    for task_entry in overview["tasks"]:
        assert len(task_entry["runs"]) == 3
        reps = [r["rep"] for r in task_entry["runs"]]
        assert reps == [1, 2, 3]
        for run in task_entry["runs"]:
            assert run["expected_session"].startswith("v104_")
            assert run["budget"] == 1000
            assert run["status"] in ("running", "finished", "stalled")


def test_v104_monitor_run_and_node_detail() -> None:
    engine = MonitorDataEngine()
    overview = engine.get_overview()
    first_task = overview["tasks"][0]["meta"]["key"]
    first_run_name = overview["tasks"][0]["runs"][0]["name"]

    detail = engine.get_run_detail(first_task, first_run_name)
    assert detail is not None
    assert detail["summary"]["task"] == first_task
    assert "nodes" in detail
    assert "recent_events" in detail
    assert "scatter_points" in detail


def test_v104_monitor_multi_version_query() -> None:
    engine = MonitorDataEngine()
    versions = engine.get_available_versions()
    version_ids = [v["id"] for v in versions]
    assert "v10_4" in version_ids
    assert "v10_3" in version_ids


def test_is_run_session_alive() -> None:
    from experiments.traceaad_v10_4.monitor import _is_run_session_alive

    active = {"v104_tsp_r1", "v104_op_r3", "v104_cvrp_r2"}
    alive, sess = _is_run_session_alive("v104_cvrp_r2", active)
    assert alive is True
    assert sess == "v104_cvrp_r2"

    alive, sess = _is_run_session_alive("v104_tsp_r2", active)
    assert alive is False
    assert sess == "v104_tsp_r2"

import pytest

from experiments.traceaad_v10_6.monitor import (
    TASKS_METADATA,
    MonitorDataEngine,
    _is_run_session_alive,
)


def test_v106_monitor_task_metadata_consistency() -> None:
    expected_tasks = {
        "tsp_construct",
        "cvrp_aco",
        "op_aco",
        "online_bin_packing",
        "vrptw_construct",
    }
    actual_tasks = {t["key"] for t in TASKS_METADATA}
    assert actual_tasks == expected_tasks


def test_v106_monitor_engine_overview_and_runs() -> None:
    engine = MonitorDataEngine()
    overview = engine.get_overview()

    assert overview["version"] == "V10.6"
    assert overview["version_id"] == "v10_6"
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
            assert run["expected_session"].startswith("v106_")
            assert run["budget"] == 1000
            assert run["status"] in ("running", "queued", "finished", "stalled")


def test_v106_monitor_run_and_node_detail() -> None:
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
    assert "v106_telemetry" in detail["summary"]
    telemetry = detail["summary"]["v106_telemetry"]
    assert "requested_operator_counts" in telemetry
    assert "fuse_fallbacks" in telemetry
    assert "parent_routes" in telemetry


def test_v106_monitor_multi_version_query() -> None:
    engine = MonitorDataEngine()
    versions = engine.get_available_versions()
    version_ids = [v["id"] for v in versions]
    assert "v10_6" in version_ids
    assert "v10_5" in version_ids
    assert "v10_4" in version_ids
    assert "v10_3" in version_ids


def test_is_run_session_alive() -> None:
    active = {"v106_tsp_r1", "v106_op_r3", "v106_cvrp_r2"}
    alive, sess = _is_run_session_alive("v106_cvrp_r2", active)
    assert alive is True
    assert sess == "v106_cvrp_r2"

    alive, sess = _is_run_session_alive("v106_tsp_r2", active)
    assert alive is False
    assert sess == "v106_tsp_r2"

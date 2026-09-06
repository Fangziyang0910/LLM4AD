import pytest

from experiments.traceaad_v10_3.monitor import (
    TASKS_METADATA,
    MonitorDataEngine,
)


def test_monitor_task_metadata_consistency() -> None:
    expected_tasks = {
        "tsp_construct",
        "cvrp_aco",
        "op_aco",
        "online_bin_packing",
        "vrptw_construct",
    }
    actual_tasks = {t["key"] for t in TASKS_METADATA}
    assert actual_tasks == expected_tasks


def test_v103_monitor_engine_overview_and_runs() -> None:
    engine = MonitorDataEngine()
    overview = engine.get_overview()

    assert overview["version"] == "V10.3"
    assert overview["version_id"] == "v10_3"
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
            assert run["expected_session"].startswith("v103_")
            assert run["budget"] == 1000
            assert run["status"] in ("running", "finished", "stalled")


def test_v103_monitor_run_and_node_detail() -> None:
    engine = MonitorDataEngine()
    detail = engine.get_run_detail("cvrp_aco", "20260904_cvrp_v103_rep1")
    assert detail is not None
    assert detail["summary"]["task"] == "cvrp_aco"
    assert detail["summary"]["rep"] == 1
    assert "nodes" in detail
    assert "recent_events" in detail
    assert "scatter_points" in detail

    if detail["best_node"] is not None:
        best_id = detail["best_node"]["id"]
        node_detail = engine.get_node_detail("cvrp_aco", "20260904_cvrp_v103_rep1", best_id)
        assert node_detail is not None
        assert node_detail["node"]["id"] == best_id
        assert "ancestors" in node_detail


def test_v103_monitor_multi_version_query() -> None:
    engine = MonitorDataEngine()
    versions = engine.get_available_versions()
    version_ids = [v["id"] for v in versions]
    assert "v10_3" in version_ids

    if "v10_2" in version_ids:
        overview_102 = engine.get_overview("v10_2")
        assert overview_102["version"] == "V10.2"
        assert overview_102["global_summary"]["total_runs"] == 15


def test_is_run_session_alive() -> None:
    from experiments.traceaad_v10_3.monitor import _is_run_session_alive

    active = {"v103_tsp_r1_r2", "v103_op_r3_r5", "v103_cvrp_r1"}
    # Exact match
    alive, sess = _is_run_session_alive("v103_cvrp_r1", active)
    assert alive is True
    assert sess == "v103_cvrp_r1"

    # Retry suffix match
    alive, sess = _is_run_session_alive("v103_tsp_r1", active)
    assert alive is True
    assert sess == "v103_tsp_r1_r2"

    alive, sess = _is_run_session_alive("v103_op_r3", active)
    assert alive is True
    assert sess == "v103_op_r3_r5"

    # Missing session
    alive, sess = _is_run_session_alive("v103_vrptw_r1", active)
    assert alive is False
    assert sess == "v103_vrptw_r1"



import json

from experiments.traceaad_v10_6.monitor import KNOWN_VERSIONS, MonitorDataEngine


def test_v107_monitor_version_and_evaluation_axis(tmp_path):
    engine = MonitorDataEngine(default_version="v10_7")
    assert engine.default_results_root == KNOWN_VERSIONS["v10_7"]["path"]
    assert engine._resolve_version_meta(None)[2:] == ("v107", "V10.7")

    run = tmp_path / "tsp_construct" / "example_rep1"
    run.mkdir(parents=True)
    (run / "run_config.json").write_text(json.dumps({"method": "v107"}))
    (run / "tree_state.json").write_text(json.dumps({
        "budget_used": 3,
        "nodes": [{"id": 0, "fitness": -10, "idea": "test idea", "code": "pass"}],
    }))
    (run / "events.jsonl").write_text(json.dumps({
        "evaluation_id": 3, "step": 0, "node_id": 0, "fitness": -10,
        "operator": "Init", "status": "ok", "prompt_tokens": 340,
    }) + "\n")
    engine = MonitorDataEngine(results_root=tmp_path, default_version="v10_7")
    overview = engine.get_overview()
    assert overview["version"] == "V10.7"
    assert overview["global_summary"]["total_evals"] == 3
    summary = overview["tasks"][0]["runs"][0]
    assert summary["expected_session"] == "v107_tsp_r1"
    assert summary["curve"] == [[3, -10]]
    detail = engine.get_run_detail("tsp_construct", "example_rep1")
    assert detail["scatter_points"][0]["step"] == 3
    assert detail["nodes"][0]["evaluation_id"] == 3
    assert detail["summary"]["v106_telemetry"]["avg_prompt_tokens"] == 340
    node = engine.get_node_detail("tsp_construct", "example_rep1", 0)["node"]
    assert node["evaluation_id"] == 3
    assert node["idea"] == "test idea"
    assert node["code"] == "pass"

import json

from experiments.traceaad_v10_11 import monitor


def _manifest(batch, suffix, created_at):
    return {
        "batch": batch,
        "created_at": created_at,
        "plan": [{"run_name": f"{batch}_{suffix}_tsp_v1011_rep1"}],
    }


def test_v1011_monitor_exposes_the_two_qwen38_batches():
    assert set(monitor.KNOWN_VERSIONS) == {
        "v10_11_q38",
        "v10_11_q38_history_code",
    }
    assert monitor.KNOWN_VERSIONS["v10_11_q38"]["is_latest"] is False
    assert monitor.KNOWN_VERSIONS["v10_11_q38_history_code"]["is_latest"] is True


def test_v1011_monitor_selects_each_batch_from_a_shared_results_root(tmp_path):
    baseline = _manifest("20260914_v1011_q38_restart2", "restart2", "2026-09-14T10:20:00")
    history = _manifest("20260914_v1011_q38_history_code", "history_code", "2026-09-14T10:50:00")
    (tmp_path / "batch_baseline.json").write_text(json.dumps(baseline), encoding="utf-8")
    (tmp_path / "batch_history.json").write_text(json.dumps(history), encoding="utf-8")

    baseline_engine = monitor.MonitorDataEngine(
        results_root=tmp_path,
        default_version="v10_11_q38",
    )
    history_engine = monitor.MonitorDataEngine(
        results_root=tmp_path,
        default_version="v10_11_q38_history_code",
    )

    assert baseline_engine._load_latest_batch_manifest(tmp_path, "v10_11_q38")["batch"] == baseline["batch"]
    assert history_engine._load_latest_batch_manifest(tmp_path, "v10_11_q38_history_code")["batch"] == history["batch"]

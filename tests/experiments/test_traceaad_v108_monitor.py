import json

from experiments.traceaad_v10_6.monitor import MonitorDataEngine


def test_monitor_uses_first_implementation_score_for_curve_and_best(tmp_path):
    run = tmp_path / 'tsp_construct' / 'example_v108_rep1'
    run.mkdir(parents=True)
    (run / 'run_config.json').write_text(json.dumps({'method': 'v108'}))
    nodes = [dict(id=i, evaluation_id=i+1, fitness=q, code=code, idea='idea')
             for i, (q, code) in enumerate([(1, 'first'), (2, 'second'), (100, 'first')])]
    (run / 'tree_state.json').write_text(json.dumps({'budget_used': 3, 'nodes': nodes}))
    engine = MonitorDataEngine(results_root=tmp_path, default_version='v10_8')
    detail = engine.get_run_detail('tsp_construct', run.name)
    assert detail['summary']['best_fitness'] == 2
    assert detail['summary']['curve'] == [[1, 1], [2, 2], [3, 2]]
    assert detail['best_node']['id'] == 1
    assert len(detail['nodes']) == 3
    assert 'v108_telemetry' in detail['summary']
    assert 'telemetry' in detail['summary']


def test_v108_monitor_launcher_session_detection(tmp_path, monkeypatch):
    from experiments.traceaad_v10_6 import monitor
    monkeypatch.setattr(monitor, '_get_active_tmux_sessions', lambda: {'v108_launcher', 'v108_tsp_r1'})
    engine = MonitorDataEngine(results_root=tmp_path, default_version='v10_8', default_session_prefix='v108')
    overview = engine.get_overview()
    assert overview['version'] == 'V10.8'
    assert overview['version_id'] == 'v10_8'
    assert overview['scheduler']['active'] is True
    assert overview['scheduler']['session'] == 'v108_launcher'


def test_v108_is_marked_latest():
    from experiments.traceaad_v10_6.monitor import KNOWN_VERSIONS
    assert KNOWN_VERSIONS['v10_8']['is_latest'] is True
    assert KNOWN_VERSIONS['v10_8']['badge'] == 'V10.8'
    assert KNOWN_VERSIONS['v10_8']['default_prefix'] == 'v108'


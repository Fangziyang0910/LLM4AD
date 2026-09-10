import json

import pytest

from experiments.traceaad_v10_6.monitor import MonitorDataEngine


@pytest.mark.parametrize("version, best, node_id", [(1080, 2, 1), (1081, 100, 2)])
def test_monitor_scores_current_nodes_and_preserves_historical_curves(tmp_path, version, best, node_id):
    run = tmp_path / 'tsp_construct' / 'example_v108_rep1'
    run.mkdir(parents=True)
    (run / 'run_config.json').write_text(json.dumps({'method': 'v108'}))
    nodes = [dict(id=i, evaluation_id=i+1, fitness=q, code=code, idea='idea')
             for i, (q, code) in enumerate([(1, 'first'), (2, 'second'), (100, 'first')])]
    (run / 'tree_state.json').write_text(json.dumps({'version': version, 'budget_used': 3, 'nodes': nodes}))
    engine = MonitorDataEngine(results_root=tmp_path, default_version='v10_8')
    detail = engine.get_run_detail('tsp_construct', run.name)
    assert detail['summary']['best_fitness'] == best
    assert detail['summary']['curve'] == [[1, 1], [2, 2], [3, best]]
    assert detail['best_node']['id'] == node_id
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


def test_v108_remains_available_after_v109_registration():
    from experiments.traceaad_v10_6.monitor import KNOWN_VERSIONS
    assert KNOWN_VERSIONS['v10_8']['is_latest'] is False
    assert KNOWN_VERSIONS['v10_8']['badge'] == 'V10.8'
    assert KNOWN_VERSIONS['v10_8']['default_prefix'] == 'v108'


def test_allocation_monitor_keeps_three_repeats_per_arm_and_queued_slots(tmp_path, monkeypatch):
    from experiments.traceaad_v10_8.launch import build_plan
    from experiments.traceaad_v10_6 import monitor
    plan = build_plan('study', 'v108alloc', allocation_study=True)
    (tmp_path / 'batch_study.json').write_text(json.dumps({'batch': 'study', 'plan': plan}))
    row = plan[0]
    run = tmp_path / row['task'] / row['run_name']
    run.mkdir(parents=True)
    (run / 'run_config.json').write_text(json.dumps({'method': 'v108', 'method_params': {'allocation_arm': 'A'}}))
    old = tmp_path / row['task'] / 'zzz_diagnostic_A_v108_rep1'
    old.mkdir()
    (old / 'run_config.json').write_text((run / 'run_config.json').read_text())
    (tmp_path / 'batch_excluded.json').write_text(json.dumps({
        'status': 'excluded_startup_diagnostic', 'plan': [{**row, 'run_name': old.name}]}))
    monkeypatch.setattr(monitor, '_get_active_tmux_sessions', lambda: {'v108alloc_launcher', row['session']})
    for arm in 'ABCD':
        engine = MonitorDataEngine(results_root=tmp_path, default_version=f'v10_8{arm.lower()}')
        overview = engine.get_overview()
        assert overview['global_summary']['total_runs'] == 3
        assert overview['scheduler']['active']
        runs = [r for task in overview['tasks'] for r in task['runs']]
        assert all(f'_{arm}_v108_rep' in r['name'] for r in runs)
        assert all(r['name'] in {item['run_name'] for item in plan} for r in runs)
        assert overview['global_summary']['running_runs'] == (1 if arm == 'A' else 0)


def test_paused_allocation_keeps_progress_and_appears_queued(tmp_path, monkeypatch):
    from experiments.traceaad_v10_6 import monitor
    from experiments.traceaad_v10_8.launch import build_plan
    plan=build_plan('paused','v108alloc',allocation_study=True)
    for r in plan:r['status']='paused'
    (tmp_path/'batch_paused.json').write_text(json.dumps({'plan':plan,'waiting_for':['formal']}))
    r=plan[0];run=tmp_path/r['task']/r['run_name'];run.mkdir(parents=True)
    (run/'run_config.json').write_text(json.dumps({'method':'v108','method_params':{'allocation_arm':'A'}}))
    (run/'tree_state.json').write_text(json.dumps({'version':1081,'budget_used':25,'nodes':[]}))
    monkeypatch.setattr(monitor,'_get_active_tmux_sessions',lambda:set())
    e=monitor.MonitorDataEngine(results_root=tmp_path,default_version='v10_8a',default_session_prefix='v108alloc')
    for _ in range(2):
        detail=e.get_run_detail(r['task'],r['run_name'])
        assert detail['summary']['status']=='queued' and detail['summary']['budget_used']==25
    monkeypatch.setattr(monitor,'_get_active_tmux_sessions',lambda:{r['session']})
    assert e.get_run_detail(r['task'],r['run_name'])['summary']['status']=='running'

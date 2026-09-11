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


def test_default_monitor_excludes_allocation_and_keeps_formal_runs(tmp_path, monkeypatch):
    from experiments.traceaad_v10_8.launch import build_plan
    from experiments.traceaad_v10_6 import monitor
    for version, metadata in monitor.KNOWN_VERSIONS.items():
        root = tmp_path / version
        root.mkdir()
        monkeypatch.setitem(metadata, 'path', root)
    root = tmp_path / 'v10_8'
    formal = build_plan('20260909_v108_formal', 'v108')
    allocation = build_plan('20260909_v108_allocation_tsp', 'v108alloc', allocation_study=True)
    (root / 'batch_formal.json').write_text(json.dumps({'batch': 'formal', 'plan': formal}))
    (root / 'batch_allocation.json').write_text(json.dumps({'batch': 'allocation', 'plan': allocation}))
    for row in [formal[0], *allocation]:
        run = root / row['task'] / row['run_name']
        run.mkdir(parents=True)
        (run / 'run_config.json').write_text(json.dumps({'method': 'v108'}))
    monkeypatch.setattr(monitor, '_get_active_tmux_sessions',
                        lambda: {'v108_launcher', 'v108alloc_launcher', formal[0]['session']})
    engine = MonitorDataEngine(default_version='v10_8')
    versions = {v['id'] for v in engine.get_available_versions()}
    assert {'v10_8', 'v10_9', 'v10_10'} <= versions
    assert versions.isdisjoint(f'v10_8{arm}' for arm in 'abcd')
    html = monitor.HTML_FILE.read_text(encoding='utf-8')
    assert all(f'value="v10_8{arm}"' not in html for arm in 'abcd')

    parsed_runs = []
    parse_summary = engine._parse_run_summary_cached

    def track_parse(run_dir, *args):
        parsed_runs.append(run_dir.name)
        return parse_summary(run_dir, *args)

    monkeypatch.setattr(engine, '_parse_run_summary_cached', track_parse)
    engine.refresh()
    assert parsed_runs == [formal[0]['run_name']]
    overview = engine.get_overview()
    assert overview['global_summary']['total_runs'] == 15
    assert overview['global_summary']['running_runs'] == 1
    assert overview['global_summary']['queued_runs'] == 14
    assert overview['scheduler']['session'] == 'v108_launcher'
    assert {r['name'] for task in overview['tasks'] for r in task['runs']} == {
        row['run_name'] for row in formal}


def test_paused_v108_keeps_progress_and_appears_queued(tmp_path, monkeypatch):
    from experiments.traceaad_v10_6 import monitor
    from experiments.traceaad_v10_8.launch import build_plan
    plan=build_plan('paused','v108')
    for r in plan:r['status']='paused'
    (tmp_path/'batch_paused.json').write_text(json.dumps({'plan':plan,'waiting_for':['formal']}))
    r=plan[0];run=tmp_path/r['task']/r['run_name'];run.mkdir(parents=True)
    (run/'run_config.json').write_text(json.dumps({'method':'v108'}))
    (run/'tree_state.json').write_text(json.dumps({'version':1081,'budget_used':25,'nodes':[]}))
    monkeypatch.setattr(monitor,'_get_active_tmux_sessions',lambda:set())
    e=monitor.MonitorDataEngine(results_root=tmp_path,default_version='v10_8',default_session_prefix='v108')
    for _ in range(2):
        detail=e.get_run_detail(r['task'],r['run_name'])
        assert detail['summary']['status']=='queued' and detail['summary']['budget_used']==25
    monkeypatch.setattr(monitor,'_get_active_tmux_sessions',lambda:{r['session']})
    assert e.get_run_detail(r['task'],r['run_name'])['summary']['status']=='running'

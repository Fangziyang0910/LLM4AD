import json
from experiments.traceaad_v10_6 import monitor
from experiments.traceaad_v10_9.launch import build_plan


def test_v109_queued_plan_and_tune_are_visible(tmp_path, monkeypatch):
    assert monitor.KNOWN_VERSIONS['v10_9']['is_latest']
    plan = build_plan('formal', 'v109')
    (tmp_path / 'batch_formal.json').write_text(json.dumps({'plan': plan, 'batch': 'formal'}))
    row = plan[0]
    run = tmp_path / row['task'] / row['run_name']
    run.mkdir(parents=True)
    (run / 'run_config.json').write_text(json.dumps({'method': 'v109'}))
    (run / 'tree_state.json').write_text(json.dumps({'version': 1090, 'budget_used': 1,
        'nodes': [{'id': 0, 'fitness': 1, 'evaluation_id': 1, 'operator': 'Tune', 'code': 'code', 'idea': 'idea'}]}))
    (run / 'events.jsonl').write_text(json.dumps({'operator': 'Tune', 'requested_operator': 'Tune',
                                                'status': 'ok', 'evaluation_id': 1, 'fitness': 1})+'\n')
    monkeypatch.setattr(monitor, '_get_active_tmux_sessions', lambda: {'v109_launcher', row['session']})
    engine = monitor.MonitorDataEngine(results_root=tmp_path, default_version='v10_9', default_session_prefix='v109')
    overview = engine.get_overview()
    assert overview['global_summary']['total_runs'] == 15
    assert overview['scheduler']['active']
    detail = engine.get_run_detail(row['task'], row['run_name'])
    assert detail['summary']['operator_counts']['Tune'] == 1
    assert detail['summary']['curve'] == [[1, 1]]


def test_restarted_batch_excludes_superseded_run_directories(tmp_path, monkeypatch):
    plan = build_plan('new', 'v109')
    old = build_plan('old', 'v109')
    (tmp_path/'batch_new.json').write_text(json.dumps({'plan':plan,'batch':'new'}))
    (tmp_path/'batch_old.json').write_text(json.dumps({'plan':old,'batch':'old','status':'superseded'}))
    r=old[0];run=tmp_path/r['task']/r['run_name'];run.mkdir(parents=True)
    (run/'run_config.json').write_text(json.dumps({'method':'v109'}))
    (run/'tree_state.json').write_text(json.dumps({'budget_used':99,'nodes':[]}))
    monkeypatch.setattr(monitor,'_get_active_tmux_sessions',lambda:set())
    engine=monitor.MonitorDataEngine(results_root=tmp_path,default_version='v10_9',default_session_prefix='v109')
    result=engine.get_overview()
    assert result['global_summary']['total_runs']==15 and result['global_summary']['total_evals']==0
    assert all(r['name'].startswith('new_') for t in result['tasks'] for r in t['runs'])

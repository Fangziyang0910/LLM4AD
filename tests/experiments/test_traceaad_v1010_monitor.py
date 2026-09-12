import json
from experiments.traceaad_v10_6 import monitor
from experiments.traceaad_v10_10.launch import build_plan


def test_v1010_queued_plan_and_tune_are_visible(tmp_path, monkeypatch):
    assert monitor.KNOWN_VERSIONS['v10_10']['default_prefix'] == 'v1010'
    plan = build_plan('20260910_v1010_formal', 'v1010')
    (tmp_path / 'batch_formal.json').write_text(json.dumps({'plan': plan, 'batch': 'formal'}))
    row = plan[0]
    run = tmp_path / row['task'] / row['run_name']
    run.mkdir(parents=True)
    (run / 'run_config.json').write_text(json.dumps({'method': 'v1010'}))
    (run / 'tree_state.json').write_text(json.dumps({'version': 10100, 'budget_used': 1,
        'nodes': [{'id': 0, 'fitness': 1, 'evaluation_id': 1, 'operator': 'Tune', 'code': 'code', 'idea': 'idea'}]}))
    (run / 'events.jsonl').write_text(json.dumps({'operator': 'Tune', 'requested_operator': 'Tune',
                                                'status': 'ok', 'evaluation_id': 1, 'fitness': 1})+'\n')
    monkeypatch.setattr(monitor, '_get_active_tmux_sessions', lambda: {'v1010_launcher', row['session']})
    engine = monitor.MonitorDataEngine(results_root=tmp_path, default_version='v10_10', default_session_prefix='v1010')
    overview = engine.get_overview()
    assert overview['global_summary']['total_runs'] == 15
    assert overview['scheduler']['active']
    detail = engine.get_run_detail(row['task'], row['run_name'])
    assert detail['summary']['operator_counts']['Tune'] == 1
    assert detail['summary']['curve'] == [[1, 1]]


def test_restarted_batch_excludes_superseded_run_directories(tmp_path, monkeypatch):
    plan = build_plan('20260910_v1010_formal_new', 'v1010')
    old = build_plan('20260910_v1010_formal_old', 'v1010')
    (tmp_path/'batch_new.json').write_text(json.dumps({'plan':plan,'batch':'new'}))
    (tmp_path/'batch_old.json').write_text(json.dumps({'plan':old,'batch':'old','status':'superseded'}))
    r=old[0];run=tmp_path/r['task']/r['run_name'];run.mkdir(parents=True)
    (run/'run_config.json').write_text(json.dumps({'method':'v1010'}))
    (run/'tree_state.json').write_text(json.dumps({'budget_used':99,'nodes':[]}))
    monkeypatch.setattr(monitor,'_get_active_tmux_sessions',lambda:set())
    engine=monitor.MonitorDataEngine(results_root=tmp_path,default_version='v10_10',default_session_prefix='v1010')
    result=engine.get_overview()
    assert result['global_summary']['total_runs']==15 and result['global_summary']['total_evals']==0
    assert all(r['name'].startswith('20260910_v1010_formal_new_') for t in result['tasks'] for r in t['runs'])


def test_old_and_new_batches_have_separate_runs_and_sessions(tmp_path, monkeypatch):
    batches = [('v10_10', '20260910_v1010_formal', 'v1010', 91),
               ('v10_10_new', '20260911_v1010_formal', 'v1010f', 7),
               (None, '20260911_v1010_think', 'v1010t', 0)]
    for version, batch, prefix, budget in batches:
        plan = build_plan(batch, prefix)
        (tmp_path / f'batch_{batch}.json').write_text(json.dumps({'batch': batch, 'plan': plan}))
        row = plan[0]
        run = tmp_path / row['task'] / row['run_name']
        run.mkdir(parents=True)
        (run / 'run_config.json').write_text(json.dumps({'method': 'v1010'}))
        (run / 'tree_state.json').write_text(json.dumps({'budget_used': budget, 'nodes': []}))
    sessions = {'v1010f_launcher', 'v1010f_tsp_r1'}
    monkeypatch.setattr(monitor, '_get_active_tmux_sessions', lambda: sessions)
    for version, batch, prefix, budget in batches[:2]:
        engine = monitor.MonitorDataEngine(results_root=tmp_path, default_version=version)
        overview = engine.get_overview()
        assert overview['scheduler']['batch'] == batch
        assert overview['scheduler']['active'] == (prefix == 'v1010f')
        assert overview['global_summary']['total_runs'] == 15
        assert overview['global_summary']['total_evals'] == budget
        assert all(r['name'].startswith(batch + '_') for t in overview['tasks'] for r in t['runs'])
        row = build_plan(batch, prefix)[0]
        detail = engine.get_run_detail(row['task'], row['run_name'])
        assert detail['summary']['expected_session'] == prefix + '_tsp_r1'
        assert detail['summary']['status'] == ('running' if prefix == 'v1010f' else 'stalled')
        sessions.clear()
        assert engine.get_run_detail(row['task'], row['run_name'])['summary']['status'] == 'stalled'
        sessions.update({'v1010f_launcher', 'v1010f_tsp_r1'})
    (tmp_path / 'batch_20260911_v1010_formal.json').unlink()
    assert engine._load_latest_batch_manifest(tmp_path, 'v10_10_new') is None


def test_monitor_keeps_only_v106_and_later_versions():
    assert 'v10_6' in monitor.KNOWN_VERSIONS and 'v10_7' in monitor.KNOWN_VERSIONS
    for dropped in ('v10_5', 'v10_4', 'v10_3', 'v10_2', 'v10_1'):
        assert dropped not in monitor.KNOWN_VERSIONS


def test_rerun2_panel_reports_progress(tmp_path, monkeypatch):
    run = tmp_path / 'experiments' / 'vrptw_construct' / 'eoh' / '20260912_rerun2_vrptw_eoh_rep1'
    (run / 'logs').mkdir(parents=True)
    (run / 'run_config.json').write_text(json.dumps(
        {'repeat': 1, 'seed': 0, 'backend': 'server1', 'llm': {'model': 'qwen3.6-27b-awq-int4'}}))
    (run / 'logs' / 'method_state.jsonl').write_text(
        json.dumps({'method': 'eoh', 'generation': 3, 'sample_count': 42}) + '\n')
    (run / 'tmux_run.log').write_text('Current best score: -21.5\n')
    monkeypatch.setattr(monitor, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(monitor, '_get_active_tmux_sessions', lambda: {'rerun2_vrptw_eoh_r1'})
    engine = monitor.MonitorDataEngine(results_root=tmp_path, default_version='v10_10')
    data = engine.get_rerun2()
    row = data['runs'][0]
    assert row['status'] == 'running'
    assert row['samples'] == 42 and row['generation'] == 3
    assert row['best'] == -21.5 and row['backend'] == 'server1'
    assert data['summary']['running'] == 1 and data['summary']['budget'] == 1000

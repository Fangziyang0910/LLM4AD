from collections import Counter
import json
import pytest

from experiments.traceaad_v10_9 import launch
from experiments.traceaad_v10_9.run import build_parser


def test_fifteen_distinct_queued_runs_use_free_slots_on_resume():
    plan = launch.build_plan('test', 'v109')
    assert len(plan) == len({r['session'] for r in plan}) == len({r['run_name'] for r in plan}) == 15
    assert set(Counter(r['task'] for r in plan).values()) == {3}
    for task in {r['task'] for r in plan}:
        assert {r['seed'] for r in plan if r['task'] == task} == {0, 1, 2}
    assignments = launch.allocate(plan, {'local': 1, 'server1': 1})
    assert len(assignments) == 2
    for row, backend in assignments:
        row['backend'] = backend
        args = build_parser().parse_args(list(launch.launch_item(row).command())[3:])
        assert args.budget == 1000 and args.n_roots == 8
    row = assignments[0][0]
    assert launch.allocate([row], {row['backend']: 0}) == []
    other = 'server3' if row['backend'] != 'server3' else 'server1'
    assert launch.allocate([row], {other: 1}) == [(row, other)]


def test_refresh_blocks_uncertain_evaluations_and_limits_resume(monkeypatch):
    plan = launch.build_plan('test', 'v109')[:4]
    for row in plan:
        row['attempts'] = 3
    statuses = {plan[0]['run_name']: 'finished', plan[1]['run_name']: 'blocked'}
    monkeypatch.setattr(launch, 'get_summary_status', lambda p: statuses.get(p.name))
    monkeypatch.setattr(launch, 'item_is_running', lambda i: i.session == plan[2]['session'])
    launch.refresh(plan, 3)
    assert [r['status'] for r in plan] == ['finished', 'blocked', 'running', 'stopped']


def test_live_launcher_requires_frozen_source():
    with pytest.raises(ValueError, match='freeze the reviewed source'):
        launch.verify_runtime()

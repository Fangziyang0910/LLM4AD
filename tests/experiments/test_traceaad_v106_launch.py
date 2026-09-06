from collections import Counter

from experiments.traceaad_v10_6 import launch
from experiments.traceaad_v10_6.run import build_parser


def test_formal_defaults_and_fifteen_unique_runs():
    args = build_parser().parse_args(['--task', 'tsp_construct', '--backend', 'server3'])
    assert (args.budget, args.n_roots, args.output_tokens, args.max_context_tokens) == (1000, 8, 16384, 32768)
    assert (args.history_tokens, args.traj_gens, args.context_margin) == (8192, 8, 256)
    plan = launch.build_plan('testbatch', 'v106_test')
    assert len(plan) == len({r['run_name'] for r in plan}) == len({r['session'] for r in plan}) == 15
    assert Counter(r['task'] for r in plan) == {task: 3 for task in launch.TASKS}
    assert {r['seed'] for r in plan} == {0, 1, 2}


def test_allocate_never_overbooks_and_keeps_resume_backend():
    plan = launch.build_plan('testbatch', 'v106_test')
    plan[0]['backend'] = 'local'
    available = {'server1': 2, 'server3': 5, 'server3b': 6, 'local': 0}
    assignments = launch.allocate(plan, available)
    assert len(assignments) == 13
    assert plan[0] not in [r for r, _ in assignments]
    assert Counter(b for _, b in assignments) == {'server1': 2, 'server3': 5, 'server3b': 6}
    assert available['server3'] == 5 and all(r['backend'] is None for r in plan[1:])


def test_allocate_diversifies_task_repeats_when_capacity_allows():
    assignments = launch.allocate(launch.build_plan('test', 'test'),
                                  {'server1': 5, 'server3': 5, 'server3b': 5, 'local': 0})
    for task in launch.TASKS:
        assert len({backend for row, backend in assignments if row['task'] == task}) == 3


def test_refresh_does_not_restart_completed_or_unknown_evaluation_runs(monkeypatch, tmp_path):
    monkeypatch.setattr(launch, 'RESULTS_ROOT', tmp_path)
    monkeypatch.setattr(launch, 'item_is_running', lambda item: item.repeat == 3)
    monkeypatch.setattr(launch, 'get_summary_status', lambda path: 'finished' if 'rep1' in str(path) else 'blocked')
    plan = launch.build_plan('test', 'test')
    launch.refresh(plan, 5)
    assert Counter(r['status'] for r in plan) == {'finished': 5, 'blocked': 5, 'running': 5}


def test_unavailable_backend_is_retried_without_blocking_healthy_backends(monkeypatch):
    calls = []
    def check(backends):
        calls.extend(backends)
        if backends == ['server3']:
            raise RuntimeError('unavailable')
    monkeypatch.setattr(launch, 'check_backends', check)
    available = {'server1': 5, 'server3': 9, 'server3b': 9, 'local': 3}
    checked = set()
    assert launch.healthy_slots(available, checked) == {**available, 'server3': 0}
    assert available['server3'] == 9
    assert checked == {'server1', 'server3b', 'local'}
    calls.clear()
    launch.healthy_slots(available, checked)
    assert calls == ['server3']

from collections import Counter

from experiments.traceaad_v10_8.launch import build_plan, launch_item, allocate
from experiments.traceaad_v10_8.run import build_parser


def test_balanced_fifteen_run_plan_and_free_slot_capacity():
    plan = build_plan('test', 'v108')
    assert len(plan) == len({r['run_name'] for r in plan}) == 15
    assert Counter(r['backend'] for r in plan) == {'server1': 4, 'server3': 4, 'server3b': 4, 'local': 3}
    for task in {r['task'] for r in plan}:
        rows = [r for r in plan if r['task'] == task]
        assert len({r['backend'] for r in rows}) == 3
        assert sorted(r['seed'] for r in rows) == [0, 1, 2]
    for row in plan:
        args = build_parser().parse_args(list(launch_item(row).command())[3:])
        assert args.budget == 1000 and args.n_roots == 8
    available = {'server1': 2, 'server3': 2, 'server3b': 2, 'local': 1}
    assignments = allocate(plan, available)
    assert len(assignments) == 7
    assert Counter(backend for _, backend in assignments) == available


def test_allocation_study_resumes_on_any_equivalent_backend():
    plan = build_plan('study', 'v108alloc', allocation_study=True)
    assert len(plan) == len({row['session'] for row in plan}) == 12
    for seed in range(3):
        rows = [row for row in plan if row['seed'] == seed]
        assert {row['allocation_arm'] for row in rows} == set('ABCD')
        assert all(row['backend'] is None for row in rows)
    assigned = allocate(plan, {'server1': 2, 'server3': 5, 'server3b': 5})
    assert len(assigned) == 12
    assert Counter(backend for _, backend in assigned) == {'server1': 2, 'server3': 5, 'server3b': 5}
    for row, backend in assigned:
        row['backend'] = backend
        args = build_parser().parse_args(list(launch_item(row).command())[3:])
        assert args.task == 'tsp_construct' and args.allocation_arm == row['allocation_arm']
        assert args.budget == 1000 and args.n_roots == 8
        row['status'] = 'running'
    resumed = next(row for row in plan if row['backend'] == 'server1')
    resumed['status'] = 'queued'
    moved = allocate(plan, {'server1': 0, 'server3': 5, 'server3b': 5})
    assert len(moved) == 1 and moved[0][1] in ('server3', 'server3b')
    pending = allocate(plan, {'server1': 1, 'server3': 5, 'server3b': 5})
    assert len(pending) == 1

from collections import Counter

from experiments.traceaad_v10_8.launch import build_plan, launch_item, allocate
from experiments.traceaad_v10_8.run import build_parser


def test_balanced_fifteen_run_plan_and_fixed_resume_backend():
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
    assert all(row['backend'] == backend for row, backend in assignments)

from experiments.traceaad_v10_7.launch import build_plan, launch_item
from experiments.traceaad_v10_7.run import build_parser


def test_every_launched_run_receives_explicit_context_configuration():
    plan = build_plan('test', 'test_v107r', 2)
    assert len(plan) == len({row['run_name'] for row in plan}) == 15
    for row in plan:
        assert row['context_policy'] == 'task_evidence_v1'
        assert row['structure_preference'] == 0.25
        assert '_v107r_rep' in row['run_name']
        item = launch_item(row)
        args = build_parser().parse_args(['--task', row['task'], *item.extra_args])
        assert args.max_context_programs == 2
        assert args.budget == 1000 and args.n_roots == 8
        assert args.output_tokens == 16384 and args.max_context_tokens == 32768


def test_default_launch_configuration_is_two_programs():
    assert build_parser().parse_args(['--task', 'tsp_construct']).max_context_programs == 2
    assert build_plan('test', 'test_v107r')[0]['max_context_programs'] == 2

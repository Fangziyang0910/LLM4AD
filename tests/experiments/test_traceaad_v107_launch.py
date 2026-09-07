import pytest

from experiments.traceaad_v10_7.launch import build_plan, launch_item
from experiments.traceaad_v10_7.run import build_parser
from llm4ad.method.traceaad_v10_7.sampling import CONTEXT_POLICIES


@pytest.mark.parametrize('policy', CONTEXT_POLICIES)
def test_every_launched_run_receives_explicit_context_configuration(policy):
    plan = build_plan('test', 'test_v107', policy, 2)
    assert len(plan) == len({row['run_name'] for row in plan}) == 15
    for row in plan:
        item = launch_item(row)
        args = build_parser().parse_args(['--task', row['task'], *item.extra_args])
        assert args.context_policy == policy and args.max_context_programs == 2
        assert args.budget == 1000 and args.n_roots == 8
        assert args.output_tokens == 16384 and args.max_context_tokens == 32768


def test_default_policy_is_stratified():
    assert build_parser().parse_args(['--task', 'tsp_construct']).context_policy == 'sampled_trajectory_v1'

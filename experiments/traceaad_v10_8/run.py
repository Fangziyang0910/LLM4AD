"""Run V10.8 with verified formation transitions and exact-code opportunities."""

import argparse
from pathlib import Path

from experiments.infra.runner import FORMAL_BUDGET, add_common_run_args, setup_experiment_run
from llm4ad.method.traceaad_v10_8 import TraceAADV108
from llm4ad.method.traceaad_v10_8.traceaad import GROUP_POLICY
from llm4ad.method.traceaad_v10_8.trajectory import CONTEXT_POLICY, GENERATION


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=16384, default_budget=FORMAL_BUDGET)
    parser.add_argument('--n-roots', type=int, default=8)
    parser.add_argument('--context-margin', type=int, default=256)
    parser.add_argument('--max-context-tokens', type=int, default=32768)
    return parser


def main():
    args = build_parser().parse_args()
    params = {key: getattr(args, key) for key in (
        'budget', 'n_roots', 'context_margin', 'max_context_tokens', 'output_tokens')}
    params.update(traj_gens=8, history_tokens=8192, ess_fraction=0.1, ess_minimum=2)
    ctx = setup_experiment_run(
        args, method='v108', method_dir=Path(__file__).resolve().parent,
        resume_file='tree_state.json',
        method_params={**params, 'generation': GENERATION,
                       'context_policy': CONTEXT_POLICY, 'group_policy': GROUP_POLICY},
        budget_basis='Actual evaluator calls, including initialization and failed evaluations; '
                     'LLM-only failures consume no evaluator slot.',
    )
    try:
        method = TraceAADV108(evaluation=ctx.evaluation, llm=ctx.llm,
                             run_dir=ctx.run_dir, seed=args.seed, task_name=args.task, **params)
        ctx.run(method.run, header=['v108: verified formation suffix; exact-code group opportunities'])
    finally:
        ctx.llm.close()


if __name__ == '__main__':
    main()

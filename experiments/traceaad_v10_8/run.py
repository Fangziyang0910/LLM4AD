"""Run V10.8 with verified formation transitions and individual selection and input-copy rejection."""

import argparse
from pathlib import Path

from experiments.infra.runner import FORMAL_BUDGET, add_common_run_args, setup_experiment_run
from llm4ad.method.traceaad_v10_8 import TraceAADV108
from llm4ad.method.traceaad_v10_8.traceaad import SELECTION_POLICY, DEDUP_POLICY, ALLOCATION_ARMS
from llm4ad.method.traceaad_v10_8.trajectory import CONTEXT_POLICY, GENERATION


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=16384, default_budget=FORMAL_BUDGET)
    parser.add_argument('--n-roots', type=int, default=8)
    parser.add_argument('--context-margin', type=int, default=256)
    parser.add_argument('--max-context-tokens', type=int, default=32768)
    parser.add_argument('--allocation-arm', choices=ALLOCATION_ARMS, default='A')
    return parser


def main():
    args = build_parser().parse_args()
    params = {key: getattr(args, key) for key in (
        'budget', 'n_roots', 'context_margin', 'max_context_tokens', 'output_tokens', 'allocation_arm')}
    params.update(traj_gens=8, history_tokens=8192, ess_fraction=0.1, ess_minimum=2)
    ctx = setup_experiment_run(
        args, method='v108', method_dir=Path(__file__).resolve().parent,
        resume_file='tree_state.json',
        method_params={**params, 'generation': GENERATION,
                       'context_policy': CONTEXT_POLICY, 'selection_policy': SELECTION_POLICY, 'dedup_policy': DEDUP_POLICY},
        budget_basis='Actual evaluator calls, including initialization and failed evaluations; '
                     'LLM-only failures and exact parent/donor copies consume no evaluator slot.',
    )
    try:
        method = TraceAADV108(evaluation=ctx.evaluation, llm=ctx.llm,
                             run_dir=ctx.run_dir, seed=args.seed, task_name=args.task, **params)
        ctx.run(method.run, header=['v108: verified formation suffix; individual selection; reject exact input copies'])
    finally:
        ctx.llm.close()


if __name__ == '__main__':
    main()

"""Run V10.10: Quality-based search with formation history, tolerant output parsing and one error-conditioned repair."""

import argparse
from pathlib import Path

from experiments.infra.runner import FORMAL_BUDGET, add_common_run_args, setup_experiment_run
from llm4ad.method.traceaad_v10_10 import TraceAADV1010
from llm4ad.method.traceaad_v10_10.traceaad import (
    DEDUP_POLICY,
    ERROR_HANDLING,
    SELECTION_POLICY,
)
from llm4ad.method.traceaad_v10_10.trajectory import CONTEXT_POLICY, GENERATION, INITIALIZATION_POLICY


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
    params.update(traj_gens=3)
    ctx = setup_experiment_run(
        args, method='v1010', method_dir=Path(__file__).resolve().parent,
        resume_file='tree_state.json',
        method_params={**params, 'generation': GENERATION, 'max_repairs': 1,
                       'error_handling': ERROR_HANDLING,
                       'initialization_policy': INITIALIZATION_POLICY, 'context_policy': CONTEXT_POLICY, 'selection_policy': SELECTION_POLICY, 'dedup_policy': DEDUP_POLICY},
        budget_basis='Actual evaluator calls, including initialization, failures and repaired candidates; '
                     'LLM-only failures and AST-identical parent/donor copies consume no evaluator slot.',
    )
    try:
        method = TraceAADV1010(evaluation=ctx.evaluation, llm=ctx.llm,
                             run_dir=ctx.run_dir, seed=args.seed, task_name=args.task, **params)
        ctx.run(method.run, header=['v1010: ESS-8 quality parent selection; Pivot 50% uniform; '
                                    'operator-specific context; at most one error-conditioned repair'])
    finally:
        ctx.llm.close()


if __name__ == '__main__':
    main()

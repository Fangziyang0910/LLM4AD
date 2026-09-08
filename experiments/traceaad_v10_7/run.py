"""Run one repaired V10.7 search with task-specific evidence per candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.infra.runner import FORMAL_BUDGET, add_common_run_args, setup_experiment_run
from llm4ad.method.traceaad_v10_6.traceaad import OPERATOR_PROBABILITIES
from llm4ad.method.traceaad_v10_7 import TraceAADV107
from llm4ad.method.traceaad_v10_7.prompts import GENERATION
from llm4ad.method.traceaad_v10_7.sampling import CONTEXT_POLICY

METHOD = 'v107r'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=16384, default_budget=FORMAL_BUDGET)
    parser.add_argument('--n-roots', type=int, default=8)
    parser.add_argument('--max-context-programs', type=int, default=2)
    parser.add_argument('--context-margin', type=int, default=256)
    parser.add_argument('--ess-fraction', type=float, default=0.1)
    parser.add_argument('--ess-minimum', type=int, default=2)
    parser.add_argument('--max-context-tokens', type=int, default=32768)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    params = {
        key: getattr(args, key) for key in [
            'budget', 'n_roots', 'max_context_programs',
            'context_margin', 'ess_fraction', 'ess_minimum',
            'output_tokens', 'max_context_tokens',
        ]
    }
    # Fixed, not configurable: the trajectory path never renders ancestor
    # history, and the donor shortlist belongs to the retired ancestor mode.
    # traj_gens/donor_topk/history_tokens only satisfy the inherited
    # constructor; they travel to the constructor but are recorded under
    # inherited_unused in run_config, never as live method parameters.
    compat = {'traj_gens': 8, 'donor_topk': 5, 'history_tokens': 8192}
    ctx = setup_experiment_run(
        args, method=METHOD, method_dir=Path(__file__).resolve().parent,
        resume_file='tree_state.json',
        method_params={
            **params, 'inherited_unused': dict(compat),
            'operator_probabilities': OPERATOR_PROBABILITIES,
            'generation': GENERATION, 'context_policy': CONTEXT_POLICY,
        },
        budget_basis=(
            f'{args.budget} actual evaluator calls including initialization and failed '
            'evaluations; LLM-only failures consume no evaluation slot'
        ),
    )
    method = TraceAADV107(
        evaluation=ctx.evaluation, llm=ctx.llm, run_dir=ctx.run_dir,
        seed=args.seed, task_name=args.task, **params, **compat,
    )
    try:
        ctx.run(method.run, header=[
            'v107r: one-call self-contained Idea and Code; task-specific evidence; '
            'parent-first; R/P/F=0.50/0.15/0.35'
        ])
    finally:
        ctx.llm.close()


if __name__ == '__main__':
    main()

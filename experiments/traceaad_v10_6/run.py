"""Run one V10.6 search with the recommended launch configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.infra.runner import FORMAL_BUDGET, add_common_run_args, setup_experiment_run
from llm4ad.method.traceaad_v10_6 import TraceAADV106
from llm4ad.method.traceaad_v10_6.traceaad import OPERATOR_PROBABILITIES
from llm4ad.method.traceaad_v10_6.prompts import GENERATION

METHOD = "v106"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_run_args(parser, default_output_tokens=16384, default_budget=FORMAL_BUDGET)
    parser.add_argument("--n-roots", type=int, default=8)
    parser.add_argument("--donor-topk", type=int, default=5)
    parser.add_argument("--traj-gens", type=int, default=8)
    parser.add_argument("--history-tokens", type=int, default=8192)
    parser.add_argument("--summary-tokens", type=int, default=1024)
    parser.add_argument("--context-margin", type=int, default=256)
    parser.add_argument("--ess-fraction", type=float, default=0.1)
    parser.add_argument("--ess-minimum", type=int, default=2)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    params = {key: getattr(args, key) for key in [
        'summary_tokens', 'budget', 'n_roots', 'donor_topk', 'traj_gens', 'history_tokens', 'context_margin',
        'ess_fraction', 'ess_minimum', 'output_tokens', 'max_context_tokens',
    ]}
    ctx = setup_experiment_run(
        args, method=METHOD, method_dir=Path(__file__).resolve().parent,
        resume_file="tree_state.json",
        method_params={**params, 'operator_probabilities': OPERATOR_PROBABILITIES,
                       'generation': GENERATION},
        budget_basis=f"{args.budget} actual evaluator calls including initialization and failed evaluations; LLM-only failures consume no evaluation slot",
    )
    method = TraceAADV106(evaluation=ctx.evaluation, llm=ctx.llm, run_dir=ctx.run_dir,
                         seed=args.seed, task_name=args.task, **params)
    try:
        ctx.run(method.run, header=["v106: Idea then Code, independent implementation Idea; parent-first; R/P/F=0.50/0.15/0.35"])
    finally:
        ctx.llm.close()


if __name__ == "__main__":
    main()

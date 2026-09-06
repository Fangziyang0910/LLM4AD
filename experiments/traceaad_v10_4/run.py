"""Run one TraceAAD V10.4 experiment at the unified 1000-eval budget."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.infra.runner import (
    FORMAL_BUDGET,
    add_common_run_args,
    setup_experiment_run,
)
from llm4ad.method.traceaad_v10_4 import TraceAADV104

METHOD = "v104"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one TraceAAD V10.4 experiment.")
    add_common_run_args(parser, default_output_tokens=16384, default_budget=FORMAL_BUDGET)
    parser.add_argument("--idea-output-tokens", type=int, default=1024)
    parser.add_argument("--n-roots", type=int, default=8)
    parser.add_argument("--donor-topk", type=int, default=5)
    parser.add_argument("--traj-gens", type=int, default=8)
    parser.add_argument("--ess-fraction", type=float, default=0.1)
    parser.add_argument("--ess-minimum", type=int, default=2)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ctx = setup_experiment_run(
        args,
        method=METHOD,
        method_dir=Path(__file__).resolve().parent,
        resume_file="tree_state.json",
        method_params={
            "budget": args.budget,
            "idea_output_tokens": args.idea_output_tokens,
            "code_output_tokens": args.output_tokens,
            "n_roots": args.n_roots,
            "donor_topk": args.donor_topk,
            "traj_gens": args.traj_gens,
            "ess_fraction": args.ess_fraction,
            "ess_minimum": args.ess_minimum,
            "max_context_tokens": args.max_context_tokens,
        },
        budget_basis=(
            "one primary slot per formal evaluation call; each candidate uses one "
            "idea call followed by one code call; failed idea/code parsing consumes "
            f"no evaluator slot; all formal task comparisons use a fixed "
            f"{FORMAL_BUDGET}-evaluation budget"
        ),
    )
    method = TraceAADV104(
        evaluation=ctx.evaluation,
        llm=ctx.llm,
        run_dir=ctx.run_dir,
        budget=args.budget,
        idea_output_tokens=args.idea_output_tokens,
        n_roots=args.n_roots,
        donor_topk=args.donor_topk,
        traj_gens=args.traj_gens,
        ess_fraction=args.ess_fraction,
        ess_minimum=args.ess_minimum,
        output_tokens=args.output_tokens,
        max_context_tokens=args.max_context_tokens,
        seed=args.seed,
    )
    ctx.run(
        method.run,
        header=[
            f"v104: budget={args.budget}, n_roots={args.n_roots}, "
            f"donor_topk={args.donor_topk}, traj_gens={args.traj_gens}, "
            f"idea_output_tokens={args.idea_output_tokens}",
        ],
    )


if __name__ == "__main__":
    main()

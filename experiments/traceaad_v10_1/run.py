"""Run one TraceAAD V10.1 experiment at the unified 1000-eval budget."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from llm4ad.method.traceaad_v10_1 import TraceAADV101

from experiments.infra.base import (
    ALL_TASKS,
    BACKENDS,
    EXPERIMENTS_ROOT,
    build_llm_client,
    build_task,
    llm_payload,
    resolve_backend,
    run_in_tmux_log,
    set_random_seed,
    write_run_config,
)

METHOD = "v101"
FORMAL_BUDGET = 1000


def resolve_run_dir(task_root: Path, run_name: str | None) -> tuple[Path, str, bool]:
    """Create a fresh run dir, or accept an existing one for resume."""
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = task_root / run_name
    if run_dir.exists():
        if (run_dir / "tree_state.json").exists():
            return run_dir, run_name, True
        # died before the first attempt committed: start over in place
        print(f"run dir exists without resumable state, starting fresh: {run_dir}")
        return run_dir, run_name, False
    run_dir.mkdir(parents=True)
    return run_dir, run_name, False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one TraceAAD V10.1 experiment.")
    parser.add_argument("--task", choices=ALL_TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--output-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--budget", type=int, default=FORMAL_BUDGET)
    parser.add_argument("--n-roots", type=int, default=8)
    parser.add_argument("--donor-topk", type=int, default=5)
    parser.add_argument("--traj-gens", type=int, default=8)
    parser.add_argument("--ess-fraction", type=float, default=0.1)
    parser.add_argument("--ess-minimum", type=int, default=2)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--eval-workers", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = resolve_backend(args.backend, args.base_url, args.model, args.no_proxy)
    task_root = Path(__file__).resolve().parent / "results" / args.task
    run_dir, run_name, resumed = resolve_run_dir(task_root, args.run_name)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not resumed:
        evaluation, task_config = build_task(args.task, args.eval_workers)
        write_run_config(
            run_dir,
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "run_dir": str(run_dir),
                "run_name": run_name,
                "task": args.task,
                "method": METHOD,
                "repeat": args.repeat,
                "backend": args.backend,
                "seed": args.seed,
                "llm": llm_payload(
                    base_url=profile.base_url,
                    model=profile.model,
                    no_proxy=profile.no_proxy,
                    max_tokens=args.output_tokens,
                ),
                "task_eval": task_config,
                "method_params": {
                    "budget": args.budget,
                    "n_roots": args.n_roots,
                    "donor_topk": args.donor_topk,
                    "traj_gens": args.traj_gens,
                    "ess_fraction": args.ess_fraction,
                    "ess_minimum": args.ess_minimum,
                    "max_context_tokens": args.max_context_tokens,
                    "budget_basis": (
                        "one primary slot per formal evaluation call; "
                        "all formal task comparisons use a fixed "
                        f"{FORMAL_BUDGET}-evaluation budget"
                    ),
                },
            },
        )
    else:
        evaluation, _ = build_task(args.task, args.eval_workers)

    set_random_seed(args.seed)
    llm = build_llm_client(
        base_url=profile.base_url,
        model=profile.model,
        no_proxy=profile.no_proxy,
        max_tokens=args.output_tokens,
    )
    method = TraceAADV101(
        evaluation=evaluation,
        llm=llm,
        run_dir=run_dir,
        budget=args.budget,
        n_roots=args.n_roots,
        donor_topk=args.donor_topk,
        traj_gens=args.traj_gens,
        ess_fraction=args.ess_fraction,
        ess_minimum=args.ess_minimum,
        output_tokens=args.output_tokens,
        max_context_tokens=args.max_context_tokens,
        seed=args.seed,
    )
    run_in_tmux_log(
        run_dir,
        log_dir,
        [
            f"log_dir={log_dir}",
            f"llm={profile.model} @ {profile.base_url}",
            f"v101: budget={args.budget}, n_roots={args.n_roots}, "
            f"donor_topk={args.donor_topk}, traj_gens={args.traj_gens}",
            f"resumed={resumed}",
        ],
        method.run,
    )


if __name__ == "__main__":
    main()

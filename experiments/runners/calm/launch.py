"""Launch CALM (w/o GRPO) runs at the unified 1000-eval budget in tmux."""

from __future__ import annotations

import argparse

from .._common import (
    add_launch_parser_args,
    build_launch_plan,
    fill_once,
    free_slots,
    watch_and_fill,
)

MODULE = "experiments.runners.calm.run"
METHOD = "calm"
SESSION_PREFIX = "calm"
METHOD_LABEL = "CALM"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch CALM (w/o GRPO) runs at 1000-eval budget in tmux."
    )
    add_launch_parser_args(parser, watch=True, session_prefix=SESSION_PREFIX)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.repeats != 3:
        raise ValueError("paper-aligned batch uses exactly 3 repeats")
    plan = build_launch_plan(args, module=MODULE, method=METHOD)
    print(f"batch={args.batch} total={len(plan)} free={free_slots()}", flush=True)
    if args.watch:
        watch_and_fill(plan, interval_sec=args.watch_interval, dry_run=args.dry_run, method_label=METHOD_LABEL)
        return
    fill_once(plan, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

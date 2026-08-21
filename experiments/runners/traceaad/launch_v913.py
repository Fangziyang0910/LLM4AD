"""Stage A launcher: common V9.7-behavior prefix, paired control/treatment
branches.

Per task and repeat, three sequential steps (design section 9):

1. ``run-prefix``  — a V9.13 run with treatment ``pp`` from scratch to
   exactly 200 real evaluator calls.  Treatment PP never appends global
   context and the intent schedule is V9.7's, so the prefix is
   behaviorally the V9.7 protocol; using the V9.13 package keeps one
   checkpoint loader for all branches.
2. ``fork``        — copy the completed prefix directory into a control
   branch (``pp``) and a treatment branch (``fp``) and rewrite the copied
   run config to the branch configuration (budget 1000 and the branch
   treatment; checkpoint state untouched), and record a fork audit with
   hashes proving both branches restore the identical state.
3. branch runs     — both branches resume their forked checkpoint with the
   same seed, intent schedule, budget, and recovery rules to 1000 real
   evaluator calls.

The launcher drives all steps continuously: prefixes are launched into free
slots, completed prefixes are forked, and forked branches are launched as
slots free up.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .._common import (
    PRIMARY_BACKENDS,
    REPO_ROOT,
    TASKS,
    TASK_SHORT,
    free_slots,
    select_backend,
)
from . import run as run_module

PREFIX_BUDGET = 200
FULL_BUDGET = 1000


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@dataclass(frozen=True, slots=True)
class Unit:
    task: str
    repeat: int
    seed: int
    prefix_session: str
    prefix_run_name: str
    prefix_dir: Path
    ctl_dir: Path
    trt_session: str
    ctl_session: str
    trt_dir: Path


def build_plan(*, batch: str, treatment: str, session_prefix: str) -> list[Unit]:
    units: list[Unit] = []
    for repeat in range(1, 4):
        for task in TASKS:
            prefix_run_name = f"v9_13p_{batch}_{task}_rep{repeat}"
            prefix_dir = (
                REPO_ROOT / "experiments" / task / "traceaad_v9_13" / prefix_run_name
            )
            units.append(
                Unit(
                    task=task,
                    repeat=repeat,
                    seed=repeat - 1,
                    prefix_session=f"{session_prefix}_p_{TASK_SHORT[task]}_r{repeat}",
                    prefix_run_name=prefix_run_name,
                    prefix_dir=prefix_dir,
                    ctl_dir=prefix_dir.with_name(prefix_run_name + "_ctl"),
                    trt_dir=prefix_dir.with_name(prefix_run_name + f"_trt_{treatment}"),
                    trt_session=f"{session_prefix}_t_{TASK_SHORT[task]}_r{repeat}",
                    ctl_session=f"{session_prefix}_c_{TASK_SHORT[task]}_r{repeat}",
                )
            )
    return units


def _summary(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "logs" / "summary.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _running(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _prefix_done(unit: Unit) -> bool:
    summary = _summary(unit.prefix_dir)
    return bool(
        summary
        and summary.get("status") == "finished"
        and int(summary.get("evaluator_call_count", -1)) == PREFIX_BUDGET
    )


def _branch_done(run_dir: Path) -> bool:
    summary = _summary(run_dir)
    return bool(
        summary
        and summary.get("status") == "finished"
        and int(summary.get("evaluator_call_count", -1)) == FULL_BUDGET
    )


def _branch_spec(unit: Unit, *, treatment: str) -> "run_module.RunSpec":
    return run_module.make_run_spec(
        task=unit.task,
        version="v9_13",
        budget=FULL_BUDGET,
        seed=unit.seed,
        repeat=unit.repeat,
        treatment=treatment,
    )


def _state_fingerprint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "forest_sha256": hashlib.sha256(
            json.dumps(checkpoint["forest"], sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "n_candidates": checkpoint["n_candidates"],
        "n_eval": checkpoint["n_eval"],
        "iteration": checkpoint["iteration"],
        "s": checkpoint["s"],
        "pending": checkpoint["pending"] is None,
        "bootstrapped": checkpoint["bootstrapped"],
        "bootstrap_deltas": checkpoint["bootstrap_deltas"],
        "initialization_complete": checkpoint["initialization_complete"],
        "treatment_counters": checkpoint["treatment_counters"],
    }


def fork_prefix(unit: Unit, *, treatment: str) -> None:
    """Copy the completed prefix into control and treatment branch dirs."""

    if not _prefix_done(unit):
        raise ValueError(f"prefix not complete: {unit.prefix_dir}")
    checkpoint = _read_json(unit.prefix_dir / "checkpoints" / "latest.json")
    fingerprint = _state_fingerprint(checkpoint)
    branches = (
        ("control", unit.ctl_dir, "pp"),
        ("treatment", unit.trt_dir, treatment),
    )
    for role, target, branch_treatment in branches:
        if target.exists():
            raise FileExistsError(f"branch directory already exists: {target}")
        shutil.copytree(unit.prefix_dir, target, ignore=shutil.ignore_patterns("tmux_run.log"))
        branch_spec = _branch_spec(unit, treatment=branch_treatment)
        expected_params = run_module._v913_method_params(branch_spec)
        run_config = _read_json(target / "run_config.json")
        run_config["method_params"] = expected_params
        run_config["forked_from"] = unit.prefix_dir.name
        run_config["branch_role"] = role
        run_config["branch_treatment"] = branch_treatment
        run_config["prefix_evaluator_calls"] = PREFIX_BUDGET
        _write_json(target / "run_config.json", run_config)
    audit = {
        "prefix_run": unit.prefix_dir.name,
        "prefix_fingerprint": fingerprint,
        "control_dir": unit.ctl_dir.name,
        "treatment_dir": unit.trt_dir.name,
        "treatment": treatment,
        "forked_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_forest_unchanged": True,
    }
    _write_json(unit.ctl_dir.parent / f"{unit.prefix_run_name}_fork_audit.json", audit)
    print(
        f"forked {unit.prefix_dir.name} -> {unit.ctl_dir.name} (pp) + "
        f"{unit.trt_dir.name} ({treatment})",
        flush=True,
    )


def audit_fork(unit: Unit) -> dict[str, Any]:
    """Verify both branches restore the prefix state exactly."""

    audit_path = unit.ctl_dir.parent / f"{unit.prefix_run_name}_fork_audit.json"
    audit = _read_json(audit_path)
    prefix = _state_fingerprint(
        _read_json(unit.prefix_dir / "checkpoints" / "latest.json")
    )
    report: dict[str, Any] = {
        "prefix_run": unit.prefix_dir.name,
        "prefix_matches_audit": prefix == audit["prefix_fingerprint"],
        "branches": {},
    }
    for role, run_dir, treatment in (
        ("control", unit.ctl_dir, "pp"),
        ("treatment", unit.trt_dir, audit["treatment"]),
    ):
        checkpoint = _read_json(run_dir / "checkpoints" / "latest.json")
        branch = _state_fingerprint(checkpoint)
        params = _read_json(run_dir / "run_config.json")["method_params"]
        report["branches"][role] = {
            "run_dir": run_dir.name,
            "state_matches_prefix": branch == prefix,
            "budget": params.get("budget"),
            "treatment": params.get("treatment"),
            "expected_treatment": treatment,
        }
    ok = (
        report["prefix_matches_audit"]
        and all(
            row["state_matches_prefix"]
            and row["treatment"] == row["expected_treatment"]
            for row in report["branches"].values()
        )
        and report["branches"]["control"]["budget"] == FULL_BUDGET
        and report["branches"]["treatment"]["budget"] == FULL_BUDGET
    )
    report["ok"] = ok
    return report


def _launch_run(
    *,
    session: str,
    task: str,
    run_dir: Path,
    budget: int,
    seed: int,
    repeat: int,
    treatment: str,
    backend: str,
    fresh_name: str | None,
) -> None:
    command = [
        sys.executable,
        "-m",
        "experiments.runners.traceaad.run",
        "--task",
        task,
        "--version",
        "v9_13",
        "--backend",
        backend,
        "--budget",
        str(budget),
        "--repeat",
        str(repeat),
        "--seed",
        str(seed),
        "--treatment",
        treatment,
    ]
    if fresh_name is not None:
        command.extend(("--run-name", fresh_name))
    else:
        command.extend(("--resume-from", str(run_dir)))
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session, "-c", str(REPO_ROOT), *command],
        check=True,
    )


def fill_once(
    plan: list[Unit], *, treatment: str, backends: tuple[str, ...], dry_run: bool
) -> int:
    remaining = {name: free for name, free in free_slots().items() if name in backends}
    launched = 0
    for unit in plan:
        # Step 1: run the common prefix.
        if (
            not _prefix_done(unit)
            and not _running(unit.prefix_session)
            and not unit.ctl_dir.exists()
        ):
            backend = select_backend(remaining)
            if backend is None:
                break
            print(
                f"prefix task={unit.task} rep={unit.repeat} backend={backend} "
                f"session={unit.prefix_session}",
                flush=True,
            )
            if not dry_run:
                _launch_run(
                    session=unit.prefix_session,
                    task=unit.task,
                    run_dir=unit.prefix_dir,
                    budget=PREFIX_BUDGET,
                    seed=unit.seed,
                    repeat=unit.repeat,
                    treatment="pp",
                    backend=backend,
                    fresh_name=unit.prefix_run_name,
                )
            remaining[backend] -= 1
            launched += 1
            continue
        # Step 2: fork a completed prefix once.
        if _prefix_done(unit) and not unit.ctl_dir.exists():
            if dry_run:
                print(f"would fork {unit.prefix_run_name}", flush=True)
            else:
                fork_prefix(unit, treatment=treatment)
        # Step 3: run both branches.
        for role_dir, session, role_treatment in (
            (unit.ctl_dir, unit.ctl_session, "pp"),
            (unit.trt_dir, unit.trt_session, treatment),
        ):
            if _branch_done(role_dir) or _running(session) or not role_dir.exists():
                continue
            backend = select_backend(remaining)
            if backend is None:
                break
            print(
                f"branch task={unit.task} rep={unit.repeat} dir={role_dir.name} "
                f"backend={backend} session={session}",
                flush=True,
            )
            if not dry_run:
                _launch_run(
                    session=session,
                    task=unit.task,
                    run_dir=role_dir,
                    budget=FULL_BUDGET,
                    seed=unit.seed,
                    repeat=unit.repeat,
                    treatment=role_treatment,
                    backend=backend,
                    fresh_name=None,
                )
            remaining[backend] -= 1
            launched += 1
    return launched


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    launch_parser = subparsers.add_parser("launch", help="drive prefixes and branches")
    launch_parser.add_argument("--batch", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    launch_parser.add_argument(
        "--treatment", choices=("fp",), default="fp", help="Stage-A treatment arm"
    )
    launch_parser.add_argument(
        "--backend",
        choices=("server3", "server3b"),
        default=None,
        help="restrict filling to one backend (default: balance)",
    )
    launch_parser.add_argument("--session-prefix", default="v913")
    launch_parser.add_argument("--watch", action="store_true")
    launch_parser.add_argument("--watch-interval", type=int, default=120)
    launch_parser.add_argument("--dry-run", action="store_true")
    launch_parser.add_argument("--task", choices=TASKS, help="restrict to one task")

    fork_parser = subparsers.add_parser("fork", help="fork one completed prefix")
    fork_parser.add_argument("--prefix-dir", type=Path, required=True)
    fork_parser.add_argument("--treatment", choices=("fp",), default="fp")

    audit_parser = subparsers.add_parser("audit-fork", help="verify branch forks")
    audit_parser.add_argument("--prefix-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "fork":
        unit = _unit_for_prefix(args.prefix_dir, treatment=args.treatment)
        fork_prefix(unit, treatment=args.treatment)
        return
    if args.command == "audit-fork":
        audit_path = (
            args.prefix_dir.parent / f"{args.prefix_dir.name}_fork_audit.json"
        )
        treatment = _read_json(audit_path)["treatment"]
        unit = _unit_for_prefix(args.prefix_dir, treatment=treatment)
        print(json.dumps(audit_fork(unit), indent=2, sort_keys=True))
        return

    repeats = 3
    plan = build_plan(
        batch=args.batch, treatment=args.treatment, session_prefix=args.session_prefix
    )
    if args.task:
        plan = [unit for unit in plan if unit.task == args.task]
    backends = (args.backend,) if args.backend else PRIMARY_BACKENDS
    while True:
        prefixes_done = sum(_prefix_done(unit) for unit in plan)
        branches_done = sum(
            _branch_done(unit.ctl_dir) and _branch_done(unit.trt_dir) for unit in plan
        )
        print(
            f"[{datetime.now().isoformat(timespec='seconds')}] units={len(plan)} "
            f"prefixes_done={prefixes_done} pairs_done={branches_done} "
            f"free={free_slots()}",
            flush=True,
        )
        if branches_done == len(plan):
            return
        fill_once(plan, treatment=args.treatment, backends=backends, dry_run=args.dry_run)
        if not args.watch or args.dry_run:
            return
        time.sleep(args.watch_interval)


def _unit_for_prefix(prefix_dir: Path, *, treatment: str) -> Unit:
    """Reconstruct a Unit from a prefix run directory name."""

    prefix_dir = prefix_dir.resolve()
    task = prefix_dir.parent.parent.name
    name = prefix_dir.name
    if task not in TASKS or not name.startswith("v9_13p_"):
        raise ValueError(f"not a V9.13 Stage-A prefix directory: {prefix_dir}")
    repeat = int(name.rsplit("rep", 1)[1])
    batch = name[len("v9_13p_") : -len(f"_{task}_rep{repeat}")]
    short = TASK_SHORT[task]
    return Unit(
        task=task,
        repeat=repeat,
        seed=repeat - 1,
        prefix_session=f"v913_p_{short}_r{repeat}",
        prefix_run_name=name,
        prefix_dir=prefix_dir,
        ctl_dir=prefix_dir.with_name(name + "_ctl"),
        trt_dir=prefix_dir.with_name(name + f"_trt_{treatment}"),
        trt_session=f"v913_t_{short}_r{repeat}",
        ctl_session=f"v913_c_{short}_r{repeat}",
    )


if __name__ == "__main__":
    main()

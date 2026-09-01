"""Strict run discovery and batch orchestration for BehaveSim v2."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .behavesim_profiler import DEFAULT_SAMPLE_SIZE, TASKS

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO_ROOT / "experiments"
DEFAULT_OUTPUT_ROOT = EXPERIMENTS / "_logs" / "behavesim_v3"
BASELINE_METHODS = ("eoh", "reevo", "mcts_ahd", "pathwise", "calm")


@dataclass(frozen=True, slots=True)
class ProfileTarget:
    campaign: str
    task: str
    label: str
    repeat: int
    run_dir: Path
    max_edges_per_operator: int = 0


def _is_candidate_run(path: Path) -> bool:
    return any(
        candidate.exists()
        for candidate in (
            path / "checkpoints" / "latest.json",
            path / "artifacts" / "candidates.jsonl",
            path / "logs" / "samples",
        )
    )


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _configured_budget(run_dir: Path) -> int:
    config = _load_json(run_dir / "run_config.json")
    params = config.get("method_params", {})
    raw_budget = params.get("budget", params.get("max_sample_nums"))
    if raw_budget is None:
        raise RuntimeError(f"Missing configured search budget in {run_dir / 'run_config.json'}")
    return int(raw_budget)


def assert_completed_run(run_dir: Path) -> None:
    """Reject partial searches before their candidates enter a formal analysis."""
    budget = _configured_budget(run_dir)
    trace_summary_path = run_dir / "logs" / "summary.json"
    baseline_summary_path = run_dir / "logs" / "run_summary.json"
    v97_candidates_path = run_dir / "artifacts" / "candidates.jsonl"

    if trace_summary_path.exists():
        summary = _load_json(trace_summary_path)
        if summary.get("status") != "finished" or summary.get("has_pending") is not False:
            raise RuntimeError(f"Incomplete TraceAAD run: {run_dir}")
        completed_slots = summary.get("budget_slots", summary.get("evaluator_call_count"))
        if completed_slots is None or int(completed_slots) < budget:
            raise RuntimeError(
                f"TraceAAD run did not reach budget {budget}: {run_dir} ({completed_slots})"
            )
        return

    if baseline_summary_path.exists():
        summary = _load_json(baseline_summary_path)
        method = _load_json(run_dir / "run_config.json").get("method")
        completed_slots = summary.get("method_sample_count")
        if method == "calm":
            state_path = run_dir / "logs" / "method_state.jsonl"
            final_states = []
            with state_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        state = json.loads(line)
                        if state.get("phase") == "final":
                            final_states.append(state)
            completed_slots = final_states[-1].get("sample_count") if final_states else None
        abort_state_valid = method == "calm" or summary.get("search_aborted") is False
        if (
            summary.get("status") != "finished"
            or not abort_state_valid
            or completed_slots is None
            or int(completed_slots) < budget
        ):
            raise RuntimeError(
                f"Incomplete baseline run for budget {budget}: {run_dir} ({completed_slots})"
            )
        return

    if v97_candidates_path.exists():
        evaluated = 0
        with v97_candidates_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip() and json.loads(line).get("evaluator_called"):
                    evaluated += 1
        if evaluated != budget:
            raise RuntimeError(
                f"Incomplete V9.7 run for budget {budget}: {run_dir} ({evaluated})"
            )
        return

    raise RuntimeError(f"No completion evidence under {run_dir}")


def _discover_three(base_dir: Path) -> list[Path]:
    if not base_dir.is_dir():
        raise FileNotFoundError(base_dir)
    runs = sorted(
        path
        for path in base_dir.iterdir()
        if path.is_dir()
        and not path.name.startswith(("eval", "_eval"))
        and _is_candidate_run(path)
    )
    if len(runs) != 3:
        raise RuntimeError(f"Expected exactly three runs under {base_dir}, found {len(runs)}")
    for run in runs:
        assert_completed_run(run)
    return runs


def traceaad_v916_targets(tasks: Sequence[str] = TASKS) -> list[ProfileTarget]:
    targets = []
    for task in tasks:
        runs = _discover_three(EXPERIMENTS / task / "traceaad_v9_16")
        targets.extend(
            ProfileTarget(
                campaign="traceaad_v916",
                task=task,
                label="traceaad_v9_16",
                repeat=repeat,
                run_dir=run_dir,
                max_edges_per_operator=(16 if task in {"op_aco", "cvrp_aco"} else 32),
            )
            for repeat, run_dir in enumerate(runs, 1)
        )
    return targets


def traceaad_version_targets(tasks: Sequence[str] = TASKS) -> list[ProfileTarget]:
    targets = []
    for task in tasks:
        version_dirs = {
            "traceaad_v9_7": EXPERIMENTS / task / "traceaad_v9_7",
            "traceaad_v9_14": EXPERIMENTS / task / "traceaad_v9_14",
            "traceaad_v9_17": EXPERIMENTS / task / "traceaad_v9_17",
        }
        for label, base_dir in version_dirs.items():
            if label == "traceaad_v9_7" and task == "vrptw_construct":
                continue
            runs = _discover_three(base_dir)
            targets.extend(
                ProfileTarget(
                    campaign="traceaad_versions",
                    task=task,
                    label=label,
                    repeat=repeat,
                    run_dir=run_dir,
                    max_edges_per_operator=0,
                )
                for repeat, run_dir in enumerate(runs, 1)
            )
    return targets


def baseline_targets(tasks: Sequence[str] = TASKS) -> list[ProfileTarget]:
    targets = []
    for task in tasks:
        for method in BASELINE_METHODS:
            runs = _discover_three(EXPERIMENTS / task / method)
            targets.extend(
                ProfileTarget(
                    campaign="external_methods",
                    task=task,
                    label=method,
                    repeat=repeat,
                    run_dir=run_dir,
                )
                for repeat, run_dir in enumerate(runs, 1)
            )
    return targets


def target_output_dir(root: Path, target: ProfileTarget, panel: str) -> Path:
    return (
        root
        / target.campaign
        / target.task
        / target.label
        / f"rep{target.repeat}"
        / f"panel_{panel.lower()}"
    )


def run_targets(
    targets: Iterable[ProfileTarget],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    panel: str = "A",
    workers: int = 8,
    force: bool = False,
    sample_size_override: int | None = None,
) -> None:
    target_list = list(targets)
    output_root.mkdir(parents=True, exist_ok=True)
    campaign = target_list[0].campaign if target_list else "empty"
    task_slug = "-".join(sorted({target.task for target in target_list})) or "empty"
    manifest_path = output_root / f"{campaign}_{task_slug}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            [{**asdict(target), "run_dir": str(target.run_dir)} for target in target_list],
            handle,
            indent=2,
            ensure_ascii=False,
        )
    for index, target in enumerate(target_list, 1):
        out_dir = target_output_dir(output_root, target, panel)
        summary_path = out_dir / "summary.json"
        if summary_path.exists() and not force:
            print(f"[{index}/{len(target_list)}] skip existing {summary_path}", flush=True)
            continue
        sample_size = sample_size_override or DEFAULT_SAMPLE_SIZE[target.task]
        command = [
            sys.executable,
            str(REPO_ROOT / "experiments" / "analysis" / "behavesim_profiler.py"),
            "--task",
            target.task,
            "--run-dir",
            str(target.run_dir),
            "--out-dir",
            str(out_dir),
            "--panel",
            panel,
            "--sample-size",
            str(sample_size),
            "--workers",
            str(workers),
            "--max-edges-per-operator",
            str(target.max_edges_per_operator),
            "--label",
            target.label,
            "--repeat",
            str(target.repeat),
            "--campaign",
            target.campaign,
        ]
        print(
            f"[{index}/{len(target_list)}] {target.task} {target.label} rep{target.repeat}",
            flush=True,
        )
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def parse_tasks(raw_tasks: Sequence[str] | None) -> tuple[str, ...]:
    if not raw_tasks:
        return TASKS
    unknown = sorted(set(raw_tasks) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown tasks: {', '.join(unknown)}")
    return tuple(raw_tasks)

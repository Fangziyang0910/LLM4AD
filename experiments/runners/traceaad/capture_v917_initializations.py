"""Capture the exact V9.17 post-initialization state for paired ablations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from .._common import EXPERIMENTS_ROOT, TASKS

CAPTURE_DIRNAME = "paired_initialization"


def is_fork_boundary(state: dict[str, object]) -> bool:
    """Return whether a checkpoint is the first Development decision state."""
    initial_order = state.get("initial_order")
    return bool(
        state.get("version") == "v9_17"
        and state.get("phase") == "development"
        and state.get("s_r_frozen") is True
        and state.get("cycle") == 1
        and state.get("sweep") == 1
        and isinstance(initial_order, list)
        and len(initial_order) == 8
        and state.get("initial_cursor") == len(initial_order)
        and state.get("eligible_ids") == state.get("active_ids")
        and state.get("sweep_order") == []
        and state.get("sweep_cursor") == 0
        and state.get("successful_ids") == []
        and state.get("active_block") is None
        and state.get("generation") is None
        and state.get("pending") is None
        and state.get("discovery_attempted") is False
    )


def discover_run_dirs(batch: str) -> list[Path]:
    run_dirs: list[Path] = []
    prefix = f"v9_17_{batch}_"
    for task in TASKS:
        root = EXPERIMENTS_ROOT / task / "traceaad_v9_17"
        if root.is_dir():
            run_dirs.extend(path for path in root.glob(f"{prefix}*") if path.is_dir())
    return sorted(run_dirs)


def capture_run(run_dir: Path) -> bool:
    target = run_dir / CAPTURE_DIRNAME
    if (target / "complete.json").is_file():
        return True
    checkpoint = run_dir / "checkpoints" / "latest.json"
    if not checkpoint.is_file():
        return False
    try:
        raw = checkpoint.read_bytes()
        state = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return False
    capture_mode = "exact_boundary"
    if not is_fork_boundary(state):
        state = reconstruct_initial_state(run_dir, state)
        if state is None:
            return False
        capture_mode = "reconstructed_from_process_facts"

    target.mkdir(parents=True, exist_ok=True)
    target.joinpath("latest.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    _copy_initial_evaluations(
        run_dir / "evaluations.csv",
        target / "evaluations.csv",
        max_eval=int(state["n_eval"]),
    )
    _copy_initial_events(
        run_dir / "mechanism_events.jsonl", target / "mechanism_events.jsonl"
    )
    _write_best_program(state, target / "best_program.py")
    run_config = run_dir / "run_config.json"
    if run_config.is_file():
        target.joinpath("adaptive_run_config.json").write_bytes(run_config.read_bytes())
    metadata = {
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "adaptive_run_dir": str(run_dir.resolve()),
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "budget_slots": state["n_eval"],
        "s_r": state["s_r"],
        "active_ids": state["active_ids"],
        "capture_mode": capture_mode,
    }
    target.joinpath("complete.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return True


def reconstruct_initial_state(
    run_dir: Path, current: dict[str, object]
) -> dict[str, object] | None:
    """Rebuild the fork boundary from initialization evaluations and tree facts."""
    if current.get("version") != "v9_17":
        return None
    evaluations = run_dir / "evaluations.csv"
    events = run_dir / "mechanism_events.jsonl"
    if not evaluations.is_file() or not events.is_file():
        return None
    try:
        with evaluations.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    initial_primary = [
        row
        for row in rows
        if row["attempt_kind"] == "initial"
        and row["mode"] in {"root", "initial_maturation"}
    ]
    maturation_primary = [
        row for row in initial_primary if row["mode"] == "initial_maturation"
    ]
    if len(maturation_primary) != 8 * 3:
        return None
    init_eval = max(int(row["eval_count"]) for row in initial_primary)
    initial_rows = [row for row in rows if int(row["eval_count"]) <= init_eval]
    valid_child_ids = {
        int(row["child_id"])
        for row in initial_rows
        if row["status"] == "ok" and row["child_id"]
    }

    rebuilt = json.loads(json.dumps(current))
    tree = rebuilt["tree"]
    algorithms = [
        item
        for item in tree["algorithms"]
        if item["id"] == 0 or item["id"] in valid_child_ids
    ]
    roots = sorted(
        (item for item in algorithms if item["parent_id"] == 0),
        key=lambda item: item["id"],
    )
    if len(roots) != 8:
        return None
    by_id = {item["id"]: item for item in algorithms}
    for item in algorithms:
        item["count"] = 0
        item["refine_count"] = 0
        item["explore_count"] = 0
    for row in initial_primary:
        parent_id = int(row["parent_id"])
        if parent_id == 0:
            continue
        parent = by_id[parent_id]
        parent["count"] += 1
        if row["intent"] == "refine":
            parent["refine_count"] += 1
        elif row["intent"] == "explore":
            parent["explore_count"] += 1
    tree["algorithms"] = algorithms

    maximize = bool(tree["maximize"])
    hypothesis_ids = [int(root["hypothesis_id"]) for root in roots]
    block_gains = _initial_block_gains(events)
    hypotheses: list[dict[str, object]] = []
    bootstrap_deltas: list[float] = []
    for hypothesis_id, root in zip(hypothesis_ids, roots, strict=True):
        nodes = [item for item in algorithms if item["hypothesis_id"] == hypothesis_id]
        frontier = max(
            nodes,
            key=lambda item: (
                item["fitness"] if maximize else -item["fitness"],
                -item["id"],
            ),
        )
        best_quality = frontier["fitness"] if maximize else -frontier["fitness"]
        refine_slots = sum(
            row["intent"] == "refine"
            and row["hypothesis_id"] == str(hypothesis_id)
            for row in initial_primary
        )
        hypotheses.append(
            {
                "id": hypothesis_id,
                "origin_node_id": root["id"],
                "source_hypothesis_id": None,
                "status": "active",
                "frontier_node_id": frontier["id"],
                "best_quality": best_quality,
                "primary_slots": 1 + refine_slots,
                "last_block_gain": block_gains[hypothesis_id],
            }
        )
        for node in nodes:
            if node["created_by"] != "refine":
                continue
            parent = by_id[node["parent_id"]]
            child_q = node["fitness"] if maximize else -node["fitness"]
            parent_q = parent["fitness"] if maximize else -parent["fitness"]
            bootstrap_deltas.append(abs(child_q - parent_q))

    hypotheses.sort(key=lambda item: item["id"])
    active_ids = [
        item["id"]
        for item in sorted(
            hypotheses, key=lambda item: (-item["best_quality"], item["id"])
        )
    ]
    root_primary = [row for row in initial_primary if row["mode"] == "root"]
    rebuilt.update(
        {
            "tree": tree,
            "hypotheses": hypotheses,
            "active_ids": active_ids,
            "reserve_ids": [],
            "phase": "development",
            "generation": None,
            "pending": None,
            "n_eval": init_eval,
            "n_llm_calls": len(initial_rows),
            "repair_llm_calls": sum(
                row["attempt_kind"] == "repair" for row in initial_rows
            ),
            "n_calls": len(initial_rows),
            "root_slots": len(root_primary),
            "refine_slots": len(maturation_primary),
            "explore_slots": 0,
            "next_hypothesis_id": max(hypothesis_ids) + 1,
            "next_block_id": 9,
            "initial_order": hypothesis_ids,
            "initial_cursor": 8,
            "bootstrap_deltas": bootstrap_deltas,
            "s_r": (
                float(statistics.median(bootstrap_deltas))
                if bootstrap_deltas
                else 0.0
            ),
            "s_r_frozen": True,
            "cycle": 1,
            "sweep": 1,
            "eligible_ids": active_ids,
            "sweep_order": [],
            "sweep_cursor": 0,
            "successful_ids": [],
            "active_block": None,
            "terminal_after_block": False,
            "discovery_attempted": False,
            "discovery_source_id": None,
            "discovery_candidate_hypothesis_id": None,
            "maturing_hypothesis_id": None,
            "discovery_attempts": 0,
            "valid_discoveries": 0,
            "block_counts": {
                "initial_maturation": 8,
                "development": 0,
                "maturation": 0,
                "terminal": 0,
            },
        }
    )
    return rebuilt if is_fork_boundary(rebuilt) else None


def _initial_block_gains(events: Path) -> dict[int, float]:
    gains: dict[int, float] = {}
    for line in events.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            event.get("event") == "block_finish"
            and event.get("block_kind") == "initial_maturation"
        ):
            gains[int(event["hypothesis_id"])] = float(event["gain"])
    if len(gains) != 8:
        raise RuntimeError(f"expected eight initialization block gains in {events}")
    return gains


def _copy_initial_evaluations(source: Path, target: Path, *, max_eval: int) -> None:
    with source.open(newline="", encoding="utf-8") as source_file:
        reader = csv.DictReader(source_file)
        if reader.fieldnames is None:
            raise RuntimeError(f"evaluation CSV has no header: {source}")
        rows = [row for row in reader if int(row["eval_count"]) <= max_eval]
    with target.open("w", newline="", encoding="utf-8") as target_file:
        writer = csv.DictWriter(target_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_initial_events(source: Path, target: Path) -> None:
    kept: list[str] = []
    found_boundary = False
    for line in source.read_text(encoding="utf-8").splitlines():
        kept.append(line)
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "initial_maturation_complete":
            found_boundary = True
            break
    if not found_boundary:
        raise RuntimeError(f"initial maturation event is missing: {source}")
    target.write_text("\n".join(kept) + "\n", encoding="utf-8")


def _write_best_program(state: dict[str, object], target: Path) -> None:
    tree = state["tree"]
    assert isinstance(tree, dict)
    maximize = bool(tree["maximize"])
    algorithms = [item for item in tree["algorithms"] if item["id"] != 0]
    if not algorithms:
        return
    best = max(
        algorithms,
        key=lambda item: (
            item["fitness"] if maximize else -item["fitness"],
            -item["id"],
        ),
    )
    target.write_text(
        f"# Fitness: {best['fitness']:.6g}\n\n{best['code'].rstrip()}\n",
        encoding="utf-8",
    )


def watch(batch: str, *, poll_seconds: float, expected: int) -> None:
    captured: set[Path] = set()
    while len(captured) < expected:
        for run_dir in discover_run_dirs(batch):
            if run_dir in captured:
                continue
            if capture_run(run_dir):
                captured.add(run_dir)
                print(
                    f"captured {len(captured)}/{expected}: {run_dir}", flush=True
                )
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.01)
    parser.add_argument("--expected", type=int, default=15)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        raise ValueError("poll-seconds must be positive")
    watch(args.batch, poll_seconds=args.poll_seconds, expected=args.expected)


if __name__ == "__main__":
    main()

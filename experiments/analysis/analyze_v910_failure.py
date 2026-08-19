"""Reproduce the process-level diagnosis of TraceAAD V9.10.

The final checkpoint is enough to expose right-censoring and allocation
diffusion even after events.jsonl has been pruned from completed runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path

from llm4ad.method.traceaad_v9_10.forest import Forest
from llm4ad.method.traceaad_v9_10.schema import ActionStatus, Intent
from llm4ad.method.traceaad_v9_10.selection import score_arms

TASKS = ("tsp_construct", "cvrp_aco", "op_aco", "online_bin_packing")


def _quantile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    return sorted(values)[int(fraction * (len(values) - 1))]


def analyze(path: Path) -> dict[str, object]:
    payload = json.loads((path / "checkpoints" / "latest.json").read_text())
    forest = Forest.from_dict(payload["forest"])
    now = int(payload["iteration"])
    actions = forest.actions()
    anchors = forest.anchors()
    pending = [item for item in actions if item.status is ActionStatus.PENDING]
    settled = [item for item in actions if item.status is ActionStatus.SETTLED]
    pending_ages = [now - item.order for item in pending]
    pending_depths = {
        str(depth): sum(
            1
            for item in pending
            if (
                0
                if item.child_id is None
                else forest.window_stats(item.child_id, max_depth=3)[1]
            )
            == depth
        )
        for depth in range(4)
    }

    arms = score_arms(forest, now_order=now, seed=0)
    marginal: dict[int, float] = {item.id: 0.0 for item in anchors}
    for arm in arms:
        marginal[arm.anchor_id] += arm.omega
    anchor_entropy = -sum(
        value * math.log(value) for value in marginal.values() if value > 0.0
    )
    ordered = sorted(arms, key=lambda item: item.omega, reverse=True)
    evidence = sorted(item.evidence_mass for item in arms)
    selection_counts = [item.n_refine + item.n_explore for item in anchors]
    return {
        "run_dir": str(path),
        "evaluations": int(payload["n_eval"]),
        "responses": now,
        "programs": len(forest.programs()),
        "anchors": len(anchors),
        "actions": len(actions),
        "pending": len(pending),
        "pending_fraction": len(pending) / len(actions) if actions else 0.0,
        "pending_age_median": statistics.median(pending_ages) if pending_ages else 0,
        "pending_age_p90": _quantile(pending_ages, 0.9),
        "pending_age_gt_100_fraction": (
            sum(age > 100 for age in pending_ages) / len(pending_ages)
            if pending_ages
            else 0.0
        ),
        "pending_depths": pending_depths,
        "settled_success": sum(item.result == 1 for item in settled),
        "settled_failure": sum(item.result == 0 for item in settled),
        "never_selected_anchor_fraction": (
            sum(count == 0 for count in selection_counts) / len(selection_counts)
            if selection_counts
            else 0.0
        ),
        "max_anchor_selections": max(selection_counts, default=0),
        "effective_anchors": math.exp(anchor_entropy),
        "effective_anchor_fraction": math.exp(anchor_entropy) / len(anchors),
        "top20_arm_mass": sum(item.omega for item in ordered[:20]),
        "refine_mass": sum(item.omega for item in arms if item.intent is Intent.REFINE),
        "arms_with_evidence_gt_0_1": sum(item > 0.1 for item in evidence),
        "arms_with_evidence_gt_0_5": sum(item > 0.5 for item in evidence),
        "evidence_median": statistics.median(evidence) if evidence else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="20260817_222517")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    records = []
    for task in TASKS:
        pattern = (
            f"experiments/{task}/traceaad_v9_10/"
            f"v9_10_{args.batch}_{task}_rep*/checkpoints/latest.json"
        )
        for checkpoint in sorted(glob.glob(pattern)):
            run_dir = Path(checkpoint).parent.parent
            record = analyze(run_dir)
            record["task"] = task
            records.append(record)
    if args.json:
        print(json.dumps(records, ensure_ascii=False, indent=2))
        return
    for record in records:
        print(
            f"{record['task']} {Path(record['run_dir']).name}: "
            f"pending={record['pending_fraction']:.3f}, "
            f"old>100={record['pending_age_gt_100_fraction']:.3f}, "
            f"never-selected={record['never_selected_anchor_fraction']:.3f}, "
            f"effective-anchors={record['effective_anchor_fraction']:.3f}, "
            f"top20={record['top20_arm_mass']:.4f}, "
            f"refine={record['refine_mass']:.3f}"
        )


if __name__ == "__main__":
    main()

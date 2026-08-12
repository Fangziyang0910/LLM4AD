"""Measure how large the local history pool actually is at each V9.5 anchor selection.

Reads `evidence_built` events from the official V9.5 batch and reports, per task, the
size of the formation pool (ancestor chain) and the direct pool (attempts from the exact
anchor state). This determines whether a top-k local history selector has any headroom:
if the pool rarely exceeds the item budget, selection is an identity function.

Read-only. Prints a summary table; does not write artifacts.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BATCH = "v9_5_20260811_171029"
DEFAULT_BUDGET = 8


def iter_evidence_events(run_dir: Path):
    decisions = run_dir / "artifacts" / "decisions.jsonl"
    if not decisions.exists():
        return
    with decisions.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "evidence_built":
                yield record


def attempt_outcomes(run_dir: Path) -> dict[int, str]:
    decisions = run_dir / "artifacts" / "decisions.jsonl"
    outcomes: dict[int, str] = {}
    if not decisions.exists():
        return outcomes
    with decisions.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") == "attempt_finalized":
                outcomes[record["attempt_id"]] = record.get("direct_outcome") or "none"
    return outcomes


def find_runs(batch: str) -> dict[str, list[Path]]:
    runs: dict[str, list[Path]] = {}
    for run_dir in sorted(REPO_ROOT.glob(f"experiments/*/traceaad_v9_5/{batch}_*")):
        task = run_dir.parent.parent.name
        runs.setdefault(task, []).append(run_dir)
    return runs


def quantile(values: list[int], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize(task: str, run_dirs: list[Path], budget: int) -> dict[str, object]:
    formation: list[int] = []
    direct: list[int] = []
    total: list[int] = []
    selected_formation: list[int] = []
    selected_direct: list[int] = []
    pool_outcomes: Counter[str] = Counter()
    shown_outcomes: Counter[str] = Counter()
    for run_dir in run_dirs:
        outcomes = attempt_outcomes(run_dir)
        for event in iter_evidence_events(run_dir):
            f_pool = len(event["formation_pool_ids"])
            d_pool = len(event["direct_pool_ids"])
            formation.append(f_pool)
            direct.append(d_pool)
            total.append(f_pool + d_pool)
            selected_formation.append(len(event["selected_formation_ids"]))
            selected_direct.append(len(event["selected_direct_ids"]))
            for attempt_id in event["formation_pool_ids"] + event["direct_pool_ids"]:
                pool_outcomes[outcomes.get(attempt_id, "unknown")] += 1
            for attempt_id in (
                event["selected_formation_ids"] + event["selected_direct_ids"]
            ):
                shown_outcomes[outcomes.get(attempt_id, "unknown")] += 1
    events = len(total)
    binding = sum(1 for value in total if value > budget)
    return {
        "task": task,
        "runs": len(run_dirs),
        "events": events,
        "formation_mean": statistics.mean(formation) if formation else 0.0,
        "formation_median": statistics.median(formation) if formation else 0.0,
        "formation_p90": quantile(formation, 0.9),
        "formation_max": max(formation) if formation else 0,
        "direct_mean": statistics.mean(direct) if direct else 0.0,
        "direct_p90": quantile(direct, 0.9),
        "direct_max": max(direct) if direct else 0,
        "direct_ge1": sum(1 for value in direct if value >= 1) / events if events else 0.0,
        "direct_ge3": sum(1 for value in direct if value >= 3) / events if events else 0.0,
        "total_median": statistics.median(total) if total else 0.0,
        "binding_rate": binding / events if events else 0.0,
        "dropped_mean": (
            statistics.mean([value - budget for value in total if value > budget])
            if binding
            else 0.0
        ),
        "selected_formation_mean": (
            statistics.mean(selected_formation) if selected_formation else 0.0
        ),
        "selected_direct_mean": (
            statistics.mean(selected_direct) if selected_direct else 0.0
        ),
        "total_hist": Counter(total),
        "pool_outcomes": pool_outcomes,
        "shown_outcomes": shown_outcomes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default=DEFAULT_BATCH)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    args = parser.parse_args()

    runs = find_runs(args.batch)
    if not runs:
        raise SystemExit(f"no runs found for batch {args.batch}")

    rows = [summarize(task, dirs, args.budget) for task, dirs in sorted(runs.items())]

    print(f"batch={args.batch}  item budget={args.budget}\n")
    header = (
        f"{'task':<20}{'runs':>5}{'events':>8}"
        f"{'form mean':>11}{'form p90':>10}{'form max':>10}"
        f"{'dir mean':>10}{'dir>=1':>8}{'dir>=3':>8}"
        f"{'pool med':>10}{'pool>8':>8}{'dropped':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['task']:<20}{row['runs']:>5}{row['events']:>8}"
            f"{row['formation_mean']:>11.2f}{row['formation_p90']:>10.0f}"
            f"{row['formation_max']:>10}"
            f"{row['direct_mean']:>10.2f}{row['direct_ge1']:>8.1%}{row['direct_ge3']:>8.1%}"
            f"{row['total_median']:>10.0f}{row['binding_rate']:>8.1%}"
            f"{row['dropped_mean']:>9.1f}"
        )

    print("\nselected composition (formation / direct), mean per prompt:")
    for row in rows:
        print(
            f"  {row['task']:<20}"
            f"{row['selected_formation_mean']:>6.2f} / {row['selected_direct_mean']:.2f}"
        )

    print("\npool-size distribution (candidate events per pool size):")
    for row in rows:
        hist = row["total_hist"]
        buckets = {
            "0": sum(count for size, count in hist.items() if size == 0),
            "1-4": sum(count for size, count in hist.items() if 1 <= size <= 4),
            "5-8": sum(count for size, count in hist.items() if 5 <= size <= 8),
            "9-16": sum(count for size, count in hist.items() if 9 <= size <= 16),
            "17+": sum(count for size, count in hist.items() if size >= 17),
        }
        total_events = row["events"]
        parts = " ".join(
            f"{label}:{count / total_events:.0%}" for label, count in buckets.items()
        )
        print(f"  {row['task']:<20}{parts}")

    print("\noutcome mix, whole pool vs what the recency rule actually shows:")
    labels = ("improve", "plateau", "regress", "invalid", "none")
    print(f"  {'task':<20}{'where':<8}" + "".join(f"{label:>10}" for label in labels))
    for row in rows:
        for where, counter in (
            ("pool", row["pool_outcomes"]),
            ("shown", row["shown_outcomes"]),
        ):
            denominator = sum(counter.values()) or 1
            cells = "".join(
                f"{counter.get(label, 0) / denominator:>10.1%}" for label in labels
            )
            print(f"  {row['task']:<20}{where:<8}{cells}")


if __name__ == "__main__":
    main()

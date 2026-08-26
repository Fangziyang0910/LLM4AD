"""Backfill best_history.jsonl for finished TraceAAD runs from retained artifacts.

Two reconstruction paths:

- evaluations.csv + checkpoints/latest.json (V9.14-V9.17): running max over
  ok rows in file order, child_id joined to program text from the checkpoint
  tree. Only runs with a finished summary are backfilled.
- artifacts/candidates.jsonl (V9.7 formal batch, qwen38 TSP/OBP): running max
  over evaluator_called records in file order; eval_count is the cumulative
  count of evaluator_called records.

Existing best_history.jsonl files are never overwritten. Every backfill is
verified: entries strictly improving, monotonic eval_count, and the final
fitness equals the checkpoint best (path A) and the best_program.py header
when that file exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _same_score(left: float, right: float) -> bool:
    # best_program.py headers and some summaries store %.6g rounded values.
    return math.isclose(left, right, rel_tol=1e-5, abs_tol=1e-9)


def _finite(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _checkpoint_codes(checkpoint_path: Path) -> dict[int, str]:
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    return {
        int(algorithm["id"]): algorithm["code"]
        for algorithm in payload["tree"]["algorithms"]
    }


def _best_program_fitness(run_dir: Path) -> float | None:
    path = run_dir / "best_program.py"
    if not path.is_file():
        return None
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    if not first_line.startswith("# Fitness:"):
        return None
    return _finite(first_line.split(":", 1)[1])


def _running_max(candidates: list[tuple[int, float, int, str]]) -> list[dict]:
    entries: list[dict] = []
    best: float | None = None
    for eval_count, fitness, child_id, program in candidates:
        if best is None or fitness > best:
            best = fitness
            entries.append(
                {
                    "eval_count": eval_count,
                    "fitness": fitness,
                    "child_id": child_id,
                    "program": program,
                }
            )
    return entries


def backfill_from_csv(run_dir: Path) -> list[dict]:
    codes = _checkpoint_codes(run_dir / "checkpoints" / "latest.json")
    candidates: list[tuple[int, float, int, str]] = []
    with (run_dir / "evaluations.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            fitness = _finite(row.get("fitness", ""))
            if fitness is None:
                continue
            child_id = int(row["child_id"])
            if child_id not in codes:
                raise RuntimeError(
                    f"{run_dir.name}: best-row child {child_id} missing from checkpoint"
                )
            candidates.append((int(row["eval_count"]), fitness, child_id, codes[child_id]))
    entries = _running_max(candidates)

    summary = json.loads((run_dir / "logs" / "summary.json").read_text(encoding="utf-8"))
    expected = summary.get("best_score")
    if expected is None or not entries or not _same_score(entries[-1]["fitness"], float(expected)):
        raise RuntimeError(
            f"{run_dir.name}: final best {entries[-1]['fitness'] if entries else None}"
            f" != summary best_score {expected}"
        )
    return entries


def backfill_from_candidates(run_dir: Path) -> list[dict]:
    candidates: list[tuple[int, float, int, str]] = []
    eval_count = 0
    with (run_dir / "artifacts" / "candidates.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not record.get("evaluator_called"):
                continue
            eval_count += 1
            fitness = _finite(record.get("child_fitness", record.get("score")))
            if fitness is None:
                continue
            candidates.append(
                (eval_count, fitness, int(record.get("child_id", record.get("program_id"))), record["program"])
            )
    return _running_max(candidates)


def _verify(run_dir: Path, entries: list[dict]) -> None:
    if not entries:
        raise RuntimeError(f"{run_dir.name}: no entries")
    previous_count = 0
    previous_fitness = -math.inf
    for entry in entries:
        if entry["eval_count"] <= previous_count:
            raise RuntimeError(f"{run_dir.name}: eval_count not monotonic")
        if entry["fitness"] <= previous_fitness:
            raise RuntimeError(f"{run_dir.name}: fitness not strictly improving")
        if not entry["program"].strip():
            raise RuntimeError(f"{run_dir.name}: empty program")
        previous_count = entry["eval_count"]
        previous_fitness = entry["fitness"]
    header_fitness = _best_program_fitness(run_dir)
    if header_fitness is not None and not _same_score(entries[-1]["fitness"], header_fitness):
        raise RuntimeError(
            f"{run_dir.name}: final best {entries[-1]['fitness']}"
            f" != best_program.py header {header_fitness}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report: list[tuple[str, str, str]] = []
    for task_dir in sorted(REPO.joinpath("experiments").iterdir()):
        if not task_dir.is_dir() or task_dir.name in {
            "analysis",
            "plotting",
            "runners",
            "_logs",
            "generation_probe",
            "其他实验",
            "机制实验",
        }:
            continue
        for method_dir in sorted(task_dir.iterdir()):
            if not method_dir.is_dir() or not method_dir.name.startswith("traceaad"):
                continue
            for run_dir in sorted(method_dir.iterdir()):
                if not run_dir.is_dir() or run_dir.name.startswith("eval_best"):
                    continue
                history = run_dir / "best_history.jsonl"
                if history.exists():
                    report.append((run_dir.name, "skip", "already exists"))
                    continue
                try:
                    if (run_dir / "logs" / "summary.json").is_file():
                        if json.loads(
                            (run_dir / "logs" / "summary.json").read_text(encoding="utf-8")
                        ).get("status") != "finished":
                            report.append((run_dir.name, "skip", "run not finished"))
                            continue
                        if not (run_dir / "evaluations.csv").is_file() or not (
                            run_dir / "checkpoints" / "latest.json"
                        ).is_file():
                            report.append((run_dir.name, "skip", "no csv+checkpoint"))
                            continue
                        entries = backfill_from_csv(run_dir)
                        source = "csv+checkpoint"
                    elif (run_dir / "checkpoints" / "latest.json").is_file():
                        report.append((run_dir.name, "skip", "run not finished"))
                        continue
                    elif (run_dir / "artifacts" / "candidates.jsonl").is_file():
                        entries = backfill_from_candidates(run_dir)
                        source = "candidates.jsonl"
                    else:
                        report.append((run_dir.name, "skip", "no program-text source"))
                        continue
                    _verify(run_dir, entries)
                except RuntimeError as error:
                    report.append((run_dir.name, "FAIL", str(error)))
                    continue
                if not args.dry_run:
                    history.write_text(
                        "".join(
                            json.dumps(entry, ensure_ascii=False) + "\n"
                            for entry in entries
                        ),
                        encoding="utf-8",
                    )
                report.append(
                    (
                        run_dir.name,
                        "ok" + (" (dry)" if args.dry_run else ""),
                        f"{source}: {len(entries)} entries,"
                        f" first@{entries[0]['eval_count']}, final@{entries[-1]['eval_count']}"
                        f" = {entries[-1]['fitness']:.6g}",
                    )
                )

    failures = 0
    for name, status, detail in report:
        if status == "FAIL":
            failures += 1
        print(f"{status:9} {name:52} {detail}")
    print(f"\n{len(report)} runs: {failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

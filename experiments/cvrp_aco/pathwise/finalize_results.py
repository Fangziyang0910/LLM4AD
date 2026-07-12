"""Wait for the two PathWise repeats, then build the CVRP result artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUN_DIRS = [
    PROJECT_ROOT / "experiments/cvrp_aco/pathwise/20260711_115024",
    PROJECT_ROOT / "experiments/cvrp_aco/pathwise/20260711_192005",
    PROJECT_ROOT / "experiments/cvrp_aco/pathwise/20260711_192010",
]
EVAL_DIR = PROJECT_ROOT / "experiments/cvrp_aco/pathwise/eval_20260711_192250_all3"
EVAL_SCRIPT = Path(__file__).with_name("evaluate_best_on_test.py")
PLOT_SCRIPT = PROJECT_ROOT / "docs/results/figures/plot_pathwise_cvrp_aco_search.py"
RESULT_DOC = PROJECT_ROOT / "docs/results/pathwise-qwen36-27b-cvrp-aco.md"
POLL_SECONDS = 60


def _summary(run_dir: Path) -> dict:
    path = run_dir / "logs/run_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _wait_for_runs() -> None:
    while True:
        states = []
        for run_dir in RUN_DIRS:
            path = run_dir / "logs/run_summary.json"
            if not path.exists():
                states.append(f"{run_dir.name}:running")
                continue
            summary = _summary(run_dir)
            status = summary.get("status")
            if status == "aborted":
                raise RuntimeError(f"PathWise run aborted: {run_dir}")
            states.append(f"{run_dir.name}:{status}")
        print(f"[{datetime.now().isoformat(timespec='seconds')}] {' '.join(states)}", flush=True)
        if all(state.endswith(":finished") for state in states):
            return
        time.sleep(POLL_SECONDS)


def _run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _write_results_doc() -> None:
    payload = json.loads((EVAL_DIR / "results.json").read_text(encoding="utf-8"))
    run_records = payload["run_records"]
    summaries = {run["run_name"]: _summary(Path(run["run_dir"])) for run in run_records}
    first_config = json.loads((RUN_DIRS[0] / "run_config.json").read_text(encoding="utf-8"))

    lines = [
        "# PathWise + Qwen3.6-27B on CVRP-ACO",
        "",
        f"Generated: `{payload['created_at']}`",
        "",
        "## Experiment data",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Method / model | `pathwise` / `qwen3.6-27b-awq` |",
        "| Task | `cvrp_aco` |",
        "| Search budget | `max_sample_nums=500` per run |",
        "| Search split | `train` (10 CVRP50 instances) |",
        "| Test splits | `test_50`, `test_100` (64 instances each) |",
        f"| ACO configuration | `n_ants={first_config['task_eval']['n_ants']}`, `n_iterations={first_config['task_eval']['n_iterations']}`, `aco_seed={first_config['task_eval']['aco_seed']}` |",
        "| Score semantics | score is negative mean best route length; higher score is better |",
        "",
        "## Search runs",
        "",
        "| Run | Status | Samples | Valid / failed | Best sample | Operator | Train best score | Artifact |",
        "|---|---|---:|---:|---:|---|---:|---|",
    ]
    for run in run_records:
        summary = summaries[run["run_name"]]
        lines.append(
            f"| {run['run_name']} | `{summary['status']}` | {summary['num_samples']} | "
            f"{summary['evaluate_success_program_num']} / {summary['evaluate_failed_program_num']} | "
            f"{run['best_sample_order']} | `{run['best_operator']}` | {run['train_best_score']:.12f} | "
            f"`experiments/cvrp_aco/pathwise/{run['run_name']}` |"
        )

    lines.extend(
        [
            "",
            "## Held-out test results",
            "",
            "Objective is mean best route length on the split; lower is better.",
            "",
            "| Split | Run | Best sample | Objective | Score | Eval seconds |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for split, split_payload in payload["results_by_split"].items():
        for row in split_payload["results"]:
            lines.append(
                f"| `{split}` | {row['run_name']} | {row['best_sample_order']} | "
                f"{row['objective']:.12f} | {row['score']:.12f} | {row['eval_seconds']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Three-run summary",
            "",
            "Mean and sample standard deviation use the three independent run objectives.",
            "",
            "| Split | Mean objective | Objective std | Mean score | Score std |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split, split_payload in payload["results_by_split"].items():
        objectives = [row["objective"] for row in split_payload["results"]]
        scores = [row["score"] for row in split_payload["results"]]
        summary = split_payload["summary"]
        score_mean = sum(scores) / len(scores)
        score_std = (sum((value - score_mean) ** 2 for value in scores) / (len(scores) - 1)) ** 0.5
        lines.append(
            f"| `{split}` | {summary['mean']:.12f} | {summary['sample_std']:.12f} | "
            f"{score_mean:.12f} | {score_std:.12f} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts and commands",
            "",
            f"- Complete evaluation artifact: `{EVAL_DIR.relative_to(PROJECT_ROOT) / 'results.json'}`",
            "- Best programs used for evaluation are stored beside `results.json`.",
            "- Evaluator: `experiments/cvrp_aco/pathwise/evaluate_best_on_test.py`.",
            "- Evaluation command:",
            "",
            "```bash",
            "uv run python experiments/cvrp_aco/pathwise/evaluate_best_on_test.py \\",
            "  experiments/cvrp_aco/pathwise/20260711_115024 \\",
            "  experiments/cvrp_aco/pathwise/20260711_192005 \\",
            "  experiments/cvrp_aco/pathwise/20260711_192010 \\",
            "  --output-dir experiments/cvrp_aco/pathwise/eval_20260711_192250_all3 \\",
            "  --splits test_50,test_100 --workers 16",
            "```",
            "",
            "## Search evolution",
            "",
            "![PathWise CVRP-ACO best-so-far training score](figures/pathwise-qwen36-27b-cvrp-aco-search-curve.png)",
            "",
            "Plot script: `docs/results/figures/plot_pathwise_cvrp_aco_search.py`.",
            "",
        ]
    )
    RESULT_DOC.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {RESULT_DOC}", flush=True)


def main() -> None:
    _wait_for_runs()
    _run(
        [
            sys.executable,
            str(EVAL_SCRIPT),
            *(str(run_dir.relative_to(PROJECT_ROOT)) for run_dir in RUN_DIRS),
            "--output-dir",
            str(EVAL_DIR.relative_to(PROJECT_ROOT)),
            "--splits",
            "test_50,test_100",
            "--workers",
            "16",
        ]
    )
    _run([sys.executable, str(PLOT_SCRIPT)])
    _write_results_doc()


if __name__ == "__main__":
    main()

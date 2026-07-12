"""Evaluate all completed CVRP-ACO repeats and write result documents."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = PROJECT_ROOT / "experiments/cvrp_aco/evaluate_best_on_test.py"
PLOTTER = PROJECT_ROOT / "docs/results/figures/plot_cvrp_aco_three_method_search.py"
EVAL_TAG = "eval_20260712_all3"
SPLITS = ("test_50", "test_100")
WORKERS = 16

METHODS: dict[str, dict[str, Any]] = {
    "mcts_ahd": {
        "label": "MCTS-AHD",
        "directory": "mcts_ahd",
        "runs": ("20260711_115024", "20260712_021911", "20260712_021957"),
        "budget": 1000,
        "result_doc": "mcts-ahd-qwen36-27b-cvrp-aco.md",
        "curve_stem": "mcts-ahd-qwen36-27b-cvrp-aco-search-curve",
    },
    "pathwise": {
        "label": "PathWise",
        "directory": "pathwise",
        "runs": ("20260711_115024", "20260711_192005", "20260711_192010"),
        "budget": 500,
        "result_doc": "pathwise-qwen36-27b-cvrp-aco.md",
        "curve_stem": "pathwise-qwen36-27b-cvrp-aco-search-curve",
    },
    "traceaad": {
        "label": "TraceAAD",
        "directory": "traceaad",
        "runs": ("20260711_115024", "20260712_041631", "20260712_041658"),
        "budget": 1000,
        "result_doc": "traceaad-qwen36-27b-cvrp-aco.md",
        "curve_stem": "traceaad-qwen36-27b-cvrp-aco-search-curve",
    },
}


def _run(command: list[str]) -> None:
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _run_dir(method: str, run_name: str) -> Path:
    return PROJECT_ROOT / "experiments/cvrp_aco" / METHODS[method]["directory"] / run_name


def _eval_dir(method: str) -> Path:
    return PROJECT_ROOT / "experiments/cvrp_aco" / METHODS[method]["directory"] / EVAL_TAG


def _summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "logs/run_summary.json").read_text(encoding="utf-8"))


def _validate_runs() -> None:
    for method, config in METHODS.items():
        for run_name in config["runs"]:
            run_dir = _run_dir(method, run_name)
            summary_path = run_dir / "logs/run_summary.json"
            if not summary_path.exists():
                raise RuntimeError(f"Missing run summary: {summary_path}")
            summary = _summary(run_dir)
            if summary.get("status") != "finished" or summary.get("search_aborted"):
                raise RuntimeError(f"Run is not a completed search: {run_dir} ({summary})")
            if summary.get("num_samples") != config["budget"]:
                raise RuntimeError(
                    f"Unexpected sample count for {run_dir}: "
                    f"{summary.get('num_samples')} != {config['budget']}"
                )


def _run_evaluations() -> None:
    for method, config in METHODS.items():
        output_dir = _eval_dir(method)
        command = [
            sys.executable,
            str(EVALUATOR),
            *(str(_run_dir(method, run_name).relative_to(PROJECT_ROOT)) for run_name in config["runs"]),
            "--output-dir",
            str(output_dir.relative_to(PROJECT_ROOT)),
            "--splits",
            ",".join(SPLITS),
            "--workers",
            str(WORKERS),
        ]
        _run(command)


def _load_payload(method: str) -> dict[str, Any]:
    path = _eval_dir(method) / "results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _ref(path: Path) -> str:
    return f"LLM4AD/{path.relative_to(PROJECT_ROOT).as_posix()}"


def _fmt(value: float | int | None, digits: int = 12) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _duration(summary: dict[str, Any]) -> str:
    seconds = summary.get("duration_seconds")
    if not isinstance(seconds, (int, float)):
        return "n/a"
    return f"{seconds / 3600:.2f} h"


def _write_method_doc(method: str, payload: dict[str, Any]) -> None:
    config = METHODS[method]
    first_config = json.loads((_run_dir(method, config["runs"][0]) / "run_config.json").read_text(encoding="utf-8"))
    task_eval = first_config.get("task_eval", {})
    method_params = first_config.get("method_params", {})
    lines = [
        f"# {config['label']} + Qwen3.6-27B on CVRP-ACO",
        "",
        f"Generated: `{payload['created_at']}`",
        "",
        "## Experiment data",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Method / model | `{method}` / `{payload['model']}` |",
        "| Task | `cvrp_aco` |",
        f"| Search budget | `max_sample_nums={config['budget']}` per run |",
        "| Search split | `train` (10 CVRP50 instances) |",
        "| Test splits | `test_50`, `test_100` (64 instances each) |",
        f"| ACO configuration | `n_ants={task_eval.get('n_ants', 30)}`, `n_iterations={task_eval.get('n_iterations', 100)}`, `aco_seed={task_eval.get('aco_seed', 1234)}` |",
        "| Score semantics | score is negative mean best route length; higher score is better |",
        "| Repeats | 3 independent completed runs |",
        "",
        "The report uses the canonical 64-instance `test_50` and `test_100` held-out splits, matching the existing CVRP-ACO result protocol. The separate `paper_test_*` splits are not mixed into this comparison.",
        "",
        "## Search runs",
        "",
        "| Run | Status | Samples | Valid / failed | Best sample | Operator | Train best score | Duration | Artifact |",
        "|---|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for row in payload["run_records"]:
        run_dir = Path(row["run_dir"])
        summary = _summary(run_dir)
        lines.append(
            f"| {row['run_name']} | `{summary['status']}` | {summary['num_samples']} | "
            f"{summary.get('evaluate_success_program_num', 'n/a')} / {summary.get('evaluate_failed_program_num', 'n/a')} | "
            f"{row['best_sample_order']} | `{row.get('best_operator')}` | {_fmt(row['train_best_score'])} | "
            f"{_duration(summary)} | `{_ref(run_dir)}` |"
        )

    lines.extend(
        [
            "",
            "## Held-out test results",
            "",
            "Objective is mean best route length on the split; lower is better.",
            "",
            "| Split | Run | Best sample | Operator | Objective | Score | Eval seconds |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for split in SPLITS:
        for row in payload["results_by_split"][split]["results"]:
            lines.append(
                f"| `{split}` | {row['run_name']} | {row['best_sample_order']} | `{row.get('best_operator')}` | "
                f"{_fmt(row['objective'])} | {_fmt(row['score'])} | {row['eval_seconds']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Three-run summary",
            "",
            "Mean and sample standard deviation use the three independent run objectives (ddof=1).",
            "",
            "| Split | Mean objective | Objective std | Mean score | Score std |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in SPLITS:
        rows = payload["results_by_split"][split]["results"]
        scores = [row["score"] for row in rows]
        score_summary = payload["results_by_split"][split]["summary"]
        score_mean = sum(scores) / len(scores)
        score_std = (sum((value - score_mean) ** 2 for value in scores) / (len(scores) - 1)) ** 0.5
        lines.append(
            f"| `{split}` | {_fmt(score_summary['mean'])} | {_fmt(score_summary['sample_std'])} | "
            f"{_fmt(score_mean)} | {_fmt(score_std)} |"
        )

    continuation = "\\"
    command_lines = [f"uv run python experiments/cvrp_aco/evaluate_best_on_test.py {continuation}"]
    command_lines.extend(
        f"  {_run_dir(method, run_name).relative_to(PROJECT_ROOT)} {continuation}"
        for run_name in config["runs"]
    )
    command_lines.append(f"  --output-dir {_eval_dir(method).relative_to(PROJECT_ROOT)} {continuation}")
    command_lines.append(f"  --splits {','.join(SPLITS)} --workers {WORKERS}")
    command = "\n".join(command_lines)
    curve = f"figures/{config['curve_stem']}.png"
    parameter_notes = [f"num_evaluators={method_params.get('num_evaluators', 'n/a')}"]
    if method_params.get("sampling_strategy") is not None:
        parameter_notes.append(f"sampling_strategy={method_params['sampling_strategy']}")
    lines.extend(
        [
            "",
            "## Artifacts and commands",
            "",
            f"- Complete evaluation artifact: `{_ref(_eval_dir(method) / 'results.json')}`",
            "- Best programs used for evaluation are stored beside `results.json`.",
            "- Shared evaluator: `experiments/cvrp_aco/evaluate_best_on_test.py`.",
            "- Evaluation command:",
            "",
            "```bash",
            command,
            "```",
            "",
            "## Search evolution",
            "",
            f"![{config['label']} CVRP-ACO best-so-far training score]({curve})",
            "",
            "The curve shows the mean best-so-far training score across the three runs; the band is the min-max range. Plot script: `docs/results/figures/plot_cvrp_aco_three_method_search.py`.",
            "For readability, the plot y-axis starts at -20; early scores below -20 are intentionally clipped.",
            "",
            f"Method parameters inherited from the run config include {', '.join(f'`{note}`' for note in parameter_notes)}.",
            "",
        ]
    )
    result_path = PROJECT_ROOT / "docs/results" / config["result_doc"]
    result_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {result_path}", flush=True)


def _write_comparison(payloads: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# CVRP-ACO Three-Method Comparison",
        "",
        f"Generated: `{datetime.now().isoformat(timespec='seconds')}`",
        "",
        "## Comparison protocol",
        "",
        "All methods use `qwen3.6-27b-awq`, the same CVRP-ACO train/test split protocol, `n_ants=30`, `n_iterations=100`, and `aco_seed=1234`. Each method has three independent search runs. The table reports the held-out route-length objective mean +/- sample std; lower is better.",
        "",
        "| Method | Search budget | test_50 objective | test_100 objective |",
        "|---|---:|---:|---:|",
    ]
    for method, config in METHODS.items():
        cells = []
        for split in SPLITS:
            summary = payloads[method]["results_by_split"][split]["summary"]
            cells.append(f"{summary['mean']:.6f} +/- {summary['sample_std']:.6f}")
        lines.append(f"| {config['label']} | {config['budget']} | {cells[0]} | {cells[1]} |")

    lines.extend(
        [
            "",
            "PathWise uses a 500-evaluation search budget, while MCTS-AHD and TraceAAD use 1000 evaluations. This is a comparison of the completed formal runs, not an equal-budget ablation.",
            "",
            "## Search evolution",
            "",
            "![CVRP-ACO three-method best-so-far training curves](figures/mcts-ahd-pathwise-traceaad-qwen36-27b-cvrp-aco-search-curve.png)",
            "",
            "The solid lines are the mean best-so-far training score across three runs; bands show the min-max range. PathWise ends at 500 evaluations, while the other methods continue to 1000.",
            "For readability, the plot y-axis starts at -20; early scores below -20 are intentionally clipped.",
            "",
            "## Result sources",
            "",
            "| Method | Authoritative result file | Evaluation artifact |",
            "|---|---|---|",
        ]
    )
    for method, config in METHODS.items():
        lines.append(
            f"| {config['label']} | `{config['result_doc']}` | `{_ref(_eval_dir(method) / 'results.json')}` |"
        )
    lines.extend(
        [
            "",
            "Evaluation scripts and run artifacts remain under `experiments/cvrp_aco/`; the method-specific pages contain every run-level test value and configuration.",
            "",
        ]
    )
    path = PROJECT_ROOT / "docs/results/cvrp-aco-qwen36-27b-method-comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}", flush=True)


def main() -> None:
    _validate_runs()
    _run_evaluations()
    _run([sys.executable, str(PLOTTER)])
    payloads = {method: _load_payload(method) for method in METHODS}
    for method, payload in payloads.items():
        _write_method_doc(method, payload)
    _write_comparison(payloads)


if __name__ == "__main__":
    main()

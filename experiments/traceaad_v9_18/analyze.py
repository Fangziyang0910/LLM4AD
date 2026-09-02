"""Audit process artifacts from the TraceAAD V9.18 A-stage runs.

The audit is descriptive.  It reads ``run_config.json``, the optional run
summary, ``evaluations.csv``, ``mechanism_events.jsonl`` and the latest
checkpoint.  Search quality is reported on the primary evaluator-slot axis;
bounded repairs are counted separately and may contribute a valid candidate
at the same primary slot.  Incomplete runs are retained in JSON, but the
Markdown output marks them as partial and excludes them from complete-run
aggregates.

Examples::

    uv run python experiments/analysis/analyze_v918_process.py \
        --experiments-root experiments \
        --json-output /tmp/v918_process.json \
        --markdown-output /tmp/v918_process.md

    uv run python experiments/analysis/analyze_v918_process.py \
        --run-dir experiments/online_bin_packing/traceaad_v9_18_q_atomic/...

Use ``--run-dir`` to audit an explicit, possibly partial run.  Without it,
the script discovers A-stage directories named ``v9_18_A0q_*`` and
``v9_18_A1o_*`` below ``--experiments-root``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
BUDGETS = (100, 250, 500, 750, 1000)
DEFAULT_BUDGET = 1000
DEFAULT_OPPORTUNITY_TAU = 2.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error, UnicodeError):
        return []


def _read_events(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Read valid JSONL events and return (events, malformed_line_count)."""

    if not path.is_file():
        return [], 0
    events: list[dict[str, Any]] = []
    malformed = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return [], 0
    for line in lines:
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if isinstance(value, dict):
            events.append(value)
        else:
            malformed += 1
    return events, malformed


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _float(value: Any, default: float | None = None) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return converted if math.isfinite(converted) else default


def _int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _truthy(value: Any) -> bool:
    return value is True or value in {1, "1", "true", "True"}


def _summary_path(run_dir: Path) -> Path | None:
    for candidate in (run_dir / "logs" / "summary.json", run_dir / "logs" / "run_summary.json"):
        if candidate.is_file():
            return candidate
    return None


def _config_budget(config: dict[str, Any], summary: dict[str, Any]) -> int:
    params = config.get("method_params")
    if isinstance(params, dict):
        value = _int(params.get("budget"))
        if value is not None and value > 0:
            return value
    value = _int(summary.get("budget_slots"))
    return value if value is not None and value > 0 else DEFAULT_BUDGET


def _run_arm(run_dir: Path, config: dict[str, Any]) -> str:
    method = str(config.get("method") or run_dir.parent.name)
    if "q_opportunity" in method or "A1o" in run_dir.name:
        return "q+O-atomic"
    if "q_atomic" in method or "A0q" in run_dir.name:
        return "q-atomic"
    return method


def _load_checkpoint(run_dir: Path) -> dict[str, Any]:
    return _read_json(run_dir / "checkpoints" / "latest.json")


def _checkpoint_algorithms(checkpoint: dict[str, Any]) -> dict[int, dict[str, Any]]:
    tree = checkpoint.get("tree")
    if not isinstance(tree, dict):
        return {}
    values = tree.get("algorithms")
    if not isinstance(values, list):
        # Older serialized snapshots used ``programs`` in the tree payload.
        values = tree.get("programs")
    result: dict[int, dict[str, Any]] = {}
    if not isinstance(values, list):
        return result
    for item in values:
        if not isinstance(item, dict):
            continue
        identifier = _int(item.get("id"))
        if identifier is not None:
            result[identifier] = item
    return result


def _valid_row(row: dict[str, str]) -> bool:
    return row.get("status") == "ok" and _finite(row.get("fitness"))


def _row_slot(row: dict[str, str]) -> int | None:
    return _int(row.get("eval_count"))


def _primary_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    # V9.18 calls the first attempt for a primary slot ``initial``.  Keeping
    # the fallback makes the audit useful for an interrupted/header-only file.
    selected = [row for row in rows if row.get("attempt_kind", "initial") == "initial"]
    return selected if selected else [row for row in rows if row.get("attempt_kind") != "repair"]


def _best_curve(
    valid_rows: Iterable[dict[str, str]],
    budgets: tuple[int, ...],
    *,
    maximize: bool,
) -> tuple[dict[str, float | None], float | None]:
    values: list[tuple[int, float]] = []
    for row in valid_rows:
        slot = _row_slot(row)
        fitness = _float(row.get("fitness"))
        if slot is not None and slot > 0 and fitness is not None:
            values.append((slot, fitness))
    best_at: dict[str, float | None] = {}
    for budget in budgets:
        eligible = [fitness for slot, fitness in values if slot <= budget]
        best_at[str(budget)] = (
            (max if maximize else min)(eligible) if eligible else None
        )
    search_best = None
    if values:
        all_fitness = [fitness for _, fitness in values]
        search_best = (max if maximize else min)(all_fitness)
    return best_at, search_best


def _probability_geometry(event: dict[str, Any]) -> dict[str, float] | None:
    raw_scores = event.get("selection_scores")
    if not isinstance(raw_scores, dict) or not raw_scores:
        return None
    scores: list[tuple[str, float]] = []
    for identifier, value in raw_scores.items():
        score = _float(value)
        if score is not None:
            scores.append((str(identifier), score))
    if not scores:
        return None
    beta = _float(event.get("beta"), 0.0) or 0.0
    peak = max(score for _, score in scores)
    weights = [
        math.exp(max(-745.0, min(0.0, beta * (score - peak))))
        for _, score in scores
    ]
    total = sum(weights)
    if total <= 0.0 or not math.isfinite(total):
        return None
    probabilities = [weight / total for weight in weights]
    entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0)
    ordered = sorted(probabilities, reverse=True)
    selected = str(event.get("selected_anchor"))
    selected_score = next((score for identifier, score in scores if identifier == selected), None)
    rank = None
    if selected_score is not None:
        rank = 1 + sum(score > selected_score for _, score in scores)
    return {
        "entropy": entropy,
        "effective_n": math.exp(entropy),
        "top5_mass": sum(ordered[:5]),
        "top10_mass": sum(ordered[:10]),
        "top20_mass": sum(ordered[:20]),
        "n_valid": float(len(scores)),
        "selected_rank": float(rank) if rank is not None else math.nan,
    }


def _mean(values: Iterable[float]) -> float | None:
    items = [value for value in values if math.isfinite(value)]
    return statistics.fmean(items) if items else None


def _min_or_none(values: Iterable[float]) -> float | None:
    items = [value for value in values if math.isfinite(value)]
    return min(items) if items else None


def _max_or_none(values: Iterable[float]) -> float | None:
    items = [value for value in values if math.isfinite(value)]
    return max(items) if items else None


def _event_geometry(events: list[dict[str, Any]], final_algorithms: dict[int, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    selected_ids: list[int] = []
    for event in events:
        if event.get("event") != "pre_decision":
            continue
        geometry = _probability_geometry(event)
        if geometry is not None:
            rows.append(geometry)
        selected = _int(event.get("selected_anchor"))
        if selected is not None:
            selected_ids.append(selected)
    selected_counts = Counter(selected_ids)
    entries = {
        identifier
        for identifier, algorithm in final_algorithms.items()
        if _truthy(algorithm.get("is_explore_entry"))
    }
    selected_entries = [identifier for identifier in selected_ids if identifier in entries]
    metrics: dict[str, Any] = {
        "decision_events": sum(event.get("event") == "pre_decision" for event in events),
        "geometry_events": len(rows),
        "selected_anchor_count": len(selected_ids),
        "selected_anchor_unique": len(selected_counts),
        "selected_anchor_top1_share": (
            max(selected_counts.values(), default=0) / len(selected_ids)
            if selected_ids
            else None
        ),
        "final_valid_anchor_count": len(
            [item for identifier, item in final_algorithms.items() if identifier != 0 and item.get("fitness") is not None]
        ),
        "final_explore_entry_count": len(entries),
        "explore_entry_selected_events": len(selected_entries),
        "explore_entry_selected_unique": len(set(selected_entries)),
        "explore_entry_selection_coverage": (
            len(set(selected_entries)) / len(entries) if entries else None
        ),
    }
    for name in ("entropy", "effective_n", "top5_mass", "top10_mass", "top20_mass", "n_valid", "selected_rank"):
        values = [item[name] for item in rows if math.isfinite(item[name])]
        metrics[f"{name}_mean"] = _mean(values)
        metrics[f"{name}_min"] = _min_or_none(values)
        metrics[f"{name}_max"] = _max_or_none(values)
    metrics["selected_anchor_top10_share"] = (
        sum(count for _, count in selected_counts.most_common(10)) / len(selected_ids)
        if selected_ids
        else None
    )
    return metrics


def _opportunity_metrics(
    events: list[dict[str, Any]],
    algorithms: dict[int, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    params = config.get("method_params")
    params = params if isinstance(params, dict) else {}
    maximize = bool(params.get("maximize", True))
    tau = _float(params.get("opportunity_tau"), DEFAULT_OPPORTUNITY_TAU) or DEFAULT_OPPORTUNITY_TAU
    configured_lambda = _float(params.get("opportunity_lambda"), 0.1) or 0.0
    lam = configured_lambda if params.get("allocation_mode") == "opportunity" else 0.0
    sigma_values = [_float(event.get("sigma_q"), 0.0) or 0.0 for event in events if event.get("event") == "pre_decision"]
    sigma = sigma_values[-1] if sigma_values else 0.0
    selected_entries: list[dict[str, float | int]] = []
    for event in events:
        if event.get("event") != "pre_decision":
            continue
        # The q-atomic control may still log an entry's derived O value for
        # diagnostics, but that value is not part of its score.  Count
        # opportunity activation only in the intervention arm.
        if event.get("allocation_mode") != "opportunity":
            continue
        selected = _int(event.get("selected_anchor"))
        if selected is None or not _truthy(algorithms.get(selected, {}).get("is_explore_entry")):
            continue
        opportunity = _float(event.get("opportunity"), 0.0) or 0.0
        after = _int(event.get("n_after"), 0) or 0
        age = max(0, after - 1)
        selected_entries.append(
            {
                "age": age,
                "opportunity": opportunity,
                "bonus": lam * sigma * opportunity,
            }
        )
    by_age: dict[str, dict[str, Any]] = {}
    grouped: dict[int, list[dict[str, float | int]]] = defaultdict(list)
    for item in selected_entries:
        grouped[int(item["age"])].append(item)
    for age, items in sorted(grouped.items()):
        values = [float(item["opportunity"]) for item in items]
        bonuses = [float(item["bonus"]) for item in items]
        expected = math.exp(-age / tau) if tau > 0 else None
        by_age[str(age)] = {
            "selected_events": len(items),
            "mean_opportunity": _mean(values),
            "mean_bonus": _mean(bonuses),
            "expected_opportunity": expected,
        }
    opportunity_values = [float(item["opportunity"]) for item in selected_entries]
    bonuses = [float(item["bonus"]) for item in selected_entries]
    nonzero = sum(value > 1e-12 for value in opportunity_values)
    bonus_nonzero = sum(value > 1e-12 for value in bonuses)
    # Older A-stage events did not persist ``parent_q``.  Fitness is immutable
    # per algorithm, so the final checkpoint is a valid fallback for the
    # selected anchor.  If neither source is available, leave the adjustment
    # unobserved instead of treating a missing q as zero.
    score_adjustments: list[float] = []
    for event in events:
        if event.get("event") != "pre_decision":
            continue
        selected_score = _float(event.get("selected_score"))
        selected = _int(event.get("selected_anchor"))
        parent_q = _float(event.get("parent_q"))
        if parent_q is None and selected is not None:
            fitness = _float(algorithms.get(selected, {}).get("fitness"))
            if fitness is not None:
                parent_q = fitness if maximize else -fitness
        if selected_score is not None and parent_q is not None:
            score_adjustments.append(selected_score - parent_q)
    score_adjustments = [value for value in score_adjustments if value > 1e-12]
    decay_errors = [
        abs(float(item["opportunity"]) - math.exp(-int(item["age"]) / tau))
        for item in selected_entries
        if tau > 0
    ]
    return {
        "tau": tau,
        "configured_lambda": configured_lambda,
        "applied_lambda": lam,
        "sigma_q": sigma,
        "entry_selection_events": len(selected_entries),
        "nonzero_opportunity_selected": nonzero,
        "nonzero_opportunity_rate": nonzero / len(selected_entries) if selected_entries else None,
        "nonzero_bonus_selected": bonus_nonzero,
        "nonzero_bonus_rate": bonus_nonzero / len(selected_entries) if selected_entries else None,
        "score_adjustment_events": len(score_adjustments),
        "score_adjustment_rate": (
            len(score_adjustments)
            / sum(event.get("event") == "pre_decision" for event in events)
            if any(event.get("event") == "pre_decision" for event in events)
            else None
        ),
        "max_score_adjustment": max(score_adjustments, default=0.0),
        "mean_selected_opportunity": _mean(opportunity_values),
        "mean_selected_bonus": _mean(bonuses),
        "max_selected_opportunity": max(opportunity_values, default=None),
        "max_selected_bonus": max(bonuses, default=None),
        "mean_abs_decay_error": _mean(decay_errors),
        "decay_by_entry_age": by_age,
    }


def _heldout_matches(run_dir: Path, heldout_root: Path | None) -> list[dict[str, Any]]:
    """Find held-out result directories whose results mention this run name."""

    roots: list[Path] = []
    if heldout_root is not None:
        roots.append(heldout_root)
    roots.append(run_dir.parent)
    candidates: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.name.startswith("eval_best") and (root / "results.json").is_file():
            candidates.add(root)
        else:
            candidates.update(path.parent for path in root.rglob("eval_best*/results.json"))
    matches: list[dict[str, Any]] = []
    for directory in sorted(candidates):
        results_path = directory / "results.json"
        payload = _read_json(results_path)
        names: set[str] = set()
        by_scale = payload.get("eval_results_by_scale")
        if isinstance(by_scale, dict):
            for scale in by_scale.values():
                if not isinstance(scale, dict):
                    continue
                result_rows = scale.get("results")
                if isinstance(result_rows, list):
                    for result in result_rows:
                        if isinstance(result, dict) and isinstance(result.get("run_name"), str):
                            names.add(result["run_name"])
        if run_dir.name in names or not names:
            matches.append(
                {
                    "path": str(directory),
                    "results_json": str(results_path),
                    "exists": directory.is_dir(),
                    "mentions_run": run_dir.name in names,
                    "scales": len(by_scale) if isinstance(by_scale, dict) else 0,
                }
            )
    return matches


def analyze_run(run_dir: Path, *, heldout_root: Path | None = None) -> dict[str, Any]:
    """Return a JSON-serializable audit record for one V9.18 run."""

    run_dir = run_dir.resolve()
    config = _read_json(run_dir / "run_config.json")
    summary_path = _summary_path(run_dir)
    summary = _read_json(summary_path) if summary_path is not None else {}
    rows = _read_rows(run_dir / "evaluations.csv")
    events, malformed_events = _read_events(run_dir / "mechanism_events.jsonl")
    checkpoint = _load_checkpoint(run_dir)
    algorithms = _checkpoint_algorithms(checkpoint)
    budget = _config_budget(config, summary)
    primary = _primary_rows(rows)
    valid_rows = [row for row in rows if _valid_row(row)]
    best_at_budget, search_best = _best_curve(valid_rows, BUDGETS, maximize=True)
    primary_slots = sorted({slot for row in primary if (slot := _row_slot(row)) is not None})
    observed_slots = max(primary_slots, default=0)
    primary_status = Counter(row.get("status", "") or "missing" for row in primary)
    all_status = Counter(row.get("status", "") or "missing" for row in rows)
    primary_valid = sum(_valid_row(row) for row in primary)
    primary_invalid = len(primary) - primary_valid
    ordinary_primary = [row for row in primary if row.get("mode") == "ordinary"]
    explore_primary = [row for row in ordinary_primary if row.get("intent") == "explore"]
    refine_primary = [row for row in ordinary_primary if row.get("intent") == "refine"]
    valid_explore = [row for row in rows if row.get("intent") == "explore" and _valid_row(row)]
    valid_explore_entries = [row for row in valid_explore if not _truthy(row.get("duplicate"))]
    valid_entry_ids = {row.get("entry_id") for row in valid_explore_entries if row.get("entry_id")}
    valid_child_ids = {row.get("child_id") for row in valid_explore_entries if row.get("child_id")}
    valid_explore_parents = {row.get("parent_id") for row in valid_explore_entries if row.get("parent_id")}
    repair_rows = [row for row in rows if row.get("attempt_kind") == "repair"]
    duplicate_rows = [row for row in rows if _truthy(row.get("duplicate"))]
    geometry = _event_geometry(events, algorithms)
    opportunity = _opportunity_metrics(events, algorithms, config)
    params = config.get("method_params") if isinstance(config.get("method_params"), dict) else {}
    summary_status = str(summary.get("status") or "")
    complete = summary_status == "finished" and _int(summary.get("budget_slots")) == budget
    status = "finished" if summary_status == "finished" else "incomplete"
    record: dict[str, Any] = {
        "run_dir": str(run_dir),
        "task": config.get("task") or run_dir.parent.parent.name,
        "repeat": config.get("repeat"),
        "method": config.get("method") or run_dir.parent.name,
        "arm": _run_arm(run_dir, config),
        "status": status,
        "complete_protocol": complete,
        "summary_status": summary_status or None,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "budget": budget,
        "observed_primary_slots": observed_slots,
        "primary_slot_count": len(primary_slots),
        "best_at_budget": best_at_budget,
        "search_best": search_best,
        "primary_counts": {
            "rows": len(primary),
            "valid": primary_valid,
            "invalid": primary_invalid,
            "status": dict(primary_status),
        },
        "all_attempt_counts": {
            "rows": len(rows),
            "valid": sum(_valid_row(row) for row in rows),
            "invalid": len(rows) - sum(_valid_row(row) for row in rows),
            "status": dict(all_status),
        },
        "repair": {
            "calls": len(repair_rows),
            "slots": len({slot for row in repair_rows if (slot := _row_slot(row)) is not None}),
            "successful_calls": sum(_valid_row(row) for row in repair_rows),
            "failed_calls": sum(not _valid_row(row) for row in repair_rows),
        },
        "duplicate": {
            "all_attempts": len(duplicate_rows),
            "primary": sum(_truthy(row.get("duplicate")) for row in primary),
            "valid": sum(_truthy(row.get("duplicate")) and _valid_row(row) for row in rows),
        },
        "operator": {
            "ordinary_primary": len(ordinary_primary),
            "explore_primary": len(explore_primary),
            "refine_primary": len(refine_primary),
            "explore_fraction": len(explore_primary) / len(ordinary_primary) if ordinary_primary else None,
            "valid_explore_attempts": len(valid_explore),
            "valid_explore_entries": len(valid_explore_entries),
            "valid_explore_entry_ids": len(valid_entry_ids),
            "valid_explore_child_ids": len(valid_child_ids),
            "valid_explore_parent_coverage": len(valid_explore_parents),
            "valid_explore_entry_rate": len(valid_explore_entries) / len(valid_explore) if valid_explore else None,
        },
        "selection_geometry": geometry,
        "opportunity": opportunity,
        "events": {
            "total": len(events),
            "pre_decision": sum(event.get("event") == "pre_decision" for event in events),
            "malformed_lines": malformed_events,
        },
        "cost": {
            "observed_evaluator_rows": len(rows),
            "summary_evaluator_call_count": summary.get("evaluator_call_count"),
            "llm_attempt_rows": len(rows),
            "primary_slots": len(primary_slots),
        },
        "heldout": _heldout_matches(run_dir, heldout_root),
        "configuration": {
            "allocation_mode": params.get("allocation_mode"),
            "explore_context": params.get("explore_context"),
            "opportunity_lambda": params.get("opportunity_lambda"),
            "opportunity_tau": params.get("opportunity_tau"),
            "max_history": params.get("max_history"),
        },
    }
    return record


def discover_runs(experiments_root: Path) -> list[Path]:
    """Discover only A-stage V9.18 run directories, excluding bootstrap runs."""

    roots = sorted(experiments_root.glob("traceaad_v9_18_q_atomic/results/*"))
    roots += sorted(experiments_root.glob("traceaad_v9_18_q_opportunity/results/*"))
    runs: list[Path] = []
    for root in roots:
        for path in sorted(root.glob("v9_18_A*_rep*")):
            if path.is_dir() and (path / "run_config.json").is_file():
                runs.append(path)
    return sorted(set(runs))


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    # Task fitness scales are incomparable (for example, TSP and OBP), so
    # never average complete runs across task families.
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("complete_protocol"):
            grouped[(str(record.get("task")), str(record.get("arm")))].append(record)
    result: dict[str, Any] = {}
    for (task, arm), items in sorted(grouped.items()):
        quality: dict[str, float | None] = {}
        for budget in BUDGETS:
            values = [record["best_at_budget"].get(str(budget)) for record in items]
            finite = [float(value) for value in values if _finite(value)]
            quality[str(budget)] = statistics.fmean(finite) if finite else None
        search = [record.get("search_best") for record in items]
        explore = [record["operator"].get("explore_fraction") for record in items]
        result[f"{task}::{arm}"] = {
            "task": task,
            "arm": arm,
            "complete_runs": len(items),
            "mean_best_at_budget": quality,
            "mean_search_best": _mean([float(value) for value in search if _finite(value)]),
            "mean_explore_fraction": _mean([float(value) for value in explore if _finite(value)]),
            "mean_selection_entropy": _mean([
                float(record["selection_geometry"]["entropy_mean"])
                for record in items
                if _finite(record["selection_geometry"].get("entropy_mean"))
            ]),
            "mean_opportunity_nonzero_rate": _mean([
                float(record["opportunity"]["nonzero_opportunity_rate"])
                for record in items
                if _finite(record["opportunity"].get("nonzero_opportunity_rate"))
            ]),
            "mean_score_adjustment_rate": _mean([
                float(record["opportunity"]["score_adjustment_rate"])
                for record in items
                if _finite(record["opportunity"].get("score_adjustment_rate"))
            ]),
        }
    return result


def build_report(run_dirs: list[Path], *, heldout_root: Path | None = None) -> dict[str, Any]:
    records = [analyze_run(path, heldout_root=heldout_root) for path in sorted(run_dirs)]
    complete = sum(record.get("complete_protocol", False) for record in records)
    return {
        "schema": "traceaad_v9_18_process_audit_v1",
        "budgets": list(BUDGETS),
        "run_count": len(records),
        "complete_run_count": complete,
        "incomplete_run_count": len(records) - complete,
        "complete_aggregates": _aggregate(records),
        "runs": records,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TraceAAD V9.18 A 阶段过程审计",
        "",
        "本页只汇总工件中已经观察到的过程。`incomplete` 运行保留用于监控，"
        "不进入完成运行聚合，也不作为质量结论。修复调用不占 primary slot，"
        "但在 repair 列单独计数。",
        "",
        f"运行数：{report.get('run_count', 0)}；完成协议运行：{report.get('complete_run_count', 0)}；"
        f"不完整运行：{report.get('incomplete_run_count', 0)}。",
        "",
        "## 运行级过程与质量",
        "",
        "| task | arm | rep | status | slots | best@100 | best@250 | best@500 | best@750 | best@1000 | search best | Explore | valid Explore entry | score bonus active | entropy | top10 | n_valid | repair | duplicate | held-out |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in report.get("runs", []):
        operator = record["operator"]
        geometry = record["selection_geometry"]
        opportunity = record["opportunity"]
        heldout = "yes" if record.get("heldout") else "no"
        best = record["best_at_budget"]
        lines.append(
            "| {task} | {arm} | {repeat} | {status} | {slots} | {b100} | {b250} | {b500} | {b750} | {b1000} | {search} | {explore} | {entries} | {opp} | {entropy} | {top10} | {nvalid} | {repair} | {duplicate} | {heldout} |".format(
                task=_fmt(record.get("task")),
                arm=_fmt(record.get("arm")),
                repeat=_fmt(record.get("repeat")),
                status="complete" if record.get("complete_protocol") else "incomplete",
                slots=_fmt(record.get("observed_primary_slots")),
                b100=_fmt(best.get("100")),
                b250=_fmt(best.get("250")),
                b500=_fmt(best.get("500")),
                b750=_fmt(best.get("750")),
                b1000=_fmt(best.get("1000")),
                search=_fmt(record.get("search_best")),
                explore=_fmt(operator.get("explore_fraction")),
                entries=_fmt(operator.get("valid_explore_entries")),
                opp=_fmt(opportunity.get("score_adjustment_rate")),
                entropy=_fmt(geometry.get("entropy_mean")),
                top10=_fmt(geometry.get("top10_mass_mean")),
                nvalid=_fmt(geometry.get("n_valid_mean")),
                repair=_fmt(record["repair"].get("calls")),
                duplicate=_fmt(record["duplicate"].get("all_attempts")),
                heldout=heldout,
            )
        )
    lines.extend(["", "## 完成运行聚合", "", "仅包含 `complete_protocol=true` 的运行；这些均值描述样本，不单独建立机制因果结论。", ""])
    aggregates = report.get("complete_aggregates", {})
    if not aggregates:
        lines.append("暂无完整运行。")
    else:
        lines.extend([
            "| task | arm | n | mean best@100 | mean best@250 | mean best@500 | mean best@750 | mean best@1000 | mean search best | mean Explore | mean entropy | score bonus active |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for _, aggregate in sorted(aggregates.items()):
            means = aggregate["mean_best_at_budget"]
            lines.append(
                f"| {aggregate['task']} | {aggregate['arm']} | {aggregate['complete_runs']} | {_fmt(means.get('100'))} | {_fmt(means.get('250'))} | {_fmt(means.get('500'))} | {_fmt(means.get('750'))} | {_fmt(means.get('1000'))} | {_fmt(aggregate.get('mean_search_best'))} | {_fmt(aggregate.get('mean_explore_fraction'))} | {_fmt(aggregate.get('mean_selection_entropy'))} | {_fmt(aggregate.get('mean_score_adjustment_rate'))} |"
            )
    lines.extend(["", "## 入口机会衰减", "", "每行按最终 checkpoint 中的 Explore 入口统计被选择时的 `n_after - 1`（选择前机会年龄）；`O` 与 bonus 分开记录。", ""])
    for record in report.get("runs", []):
        decay = record["opportunity"].get("decay_by_entry_age", {})
        if not decay:
            continue
        lines.append(f"### {record.get('task')} / {record.get('arm')} / rep{record.get('repeat')}")
        lines.append("")
        lines.append("| entry age | selected events | mean O | expected O | mean bonus |")
        lines.append("| ---: | ---: | ---: | ---: | ---: |")
        for age, values in decay.items():
            lines.append(f"| {age} | {values['selected_events']} | {_fmt(values['mean_opportunity'])} | {_fmt(values['expected_opportunity'])} | {_fmt(values['mean_bonus'])} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-root", type=Path, default=DEFAULT_EXPERIMENTS_ROOT)
    parser.add_argument("--run-dir", action="append", type=Path, help="Explicit run directory; repeat for multiple runs.")
    parser.add_argument("--heldout-root", type=Path, help="Optional root to search recursively for eval_best*/results.json.")
    parser.add_argument("--json-output", type=Path, help="Write the JSON report to this path; otherwise print JSON.")
    parser.add_argument("--markdown-output", type=Path, help="Write the Markdown report to this path.")
    args = parser.parse_args()
    run_dirs = args.run_dir or discover_runs(args.experiments_root)
    if not run_dirs:
        raise SystemExit("No V9.18 A-stage run directories found; pass --run-dir explicitly.")
    report = build_report(run_dirs, heldout_root=args.heldout_root)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.json_output:
        _write(args.json_output, encoded)
    else:
        print(encoded, end="")
    if args.markdown_output:
        _write(args.markdown_output, render_markdown(report))


if __name__ == "__main__":
    main()

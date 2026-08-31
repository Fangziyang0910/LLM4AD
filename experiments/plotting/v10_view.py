"""Read TraceAAD V10 run artifacts into the live observer payloads."""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any

TASKS = (
    "tsp_construct",
    "cvrp_aco",
    "op_aco",
    "online_bin_packing",
    "vrptw_construct",
)
TASK_LABEL = {
    "tsp_construct": "TSP 构造",
    "cvrp_aco": "CVRP-ACO",
    "op_aco": "OP-ACO",
    "online_bin_packing": "在线装箱",
    "vrptw_construct": "VRPTW 构造",
}
TASK_UNIT = {
    "tsp_construct": "路程",
    "cvrp_aco": "路程",
    "op_aco": "收益",
    "online_bin_packing": "箱数",
    "vrptw_construct": "路程",
}
MINIMIZE_TASKS = {
    "tsp_construct",
    "cvrp_aco",
    "online_bin_packing",
    "vrptw_construct",
}
RUN_NAME = re.compile(r"^v10_.+_rep(\d+)$")
OPERATORS = ("root", "develop", "pivot", "transfer", "restart", "semantic_repair")
OUTCOMES = (
    "improve",
    "plateau",
    "regress",
    "new_root",
    "duplicate",
    "invalid",
    "timeout",
)
G_HORIZONS = (1, 2, 4)
THREAD_HISTORY_KEEP = 64


def discover_runs(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.glob("*/traceaad_v10/v10_*"))
        if path.is_dir()
        and "smoke" not in path.name
        and (path / "evaluations.csv").is_file()
        and RUN_NAME.match(path.name)
    ]


def overview_payload(root: Path) -> dict[str, Any]:
    return {
        "budget": 1000,
        "runs": [overview_run(path) for path in discover_runs(root)],
    }


def overview_run(run_dir: Path) -> dict[str, Any]:
    config = _load_json(run_dir / "run_config.json")
    summary = _load_json(run_dir / "logs" / "summary.json")
    rows = slot_rows(_load_rows(run_dir / "evaluations.csv"))
    task = str(config.get("task") or run_dir.parent.parent.name)
    minimize = task in MINIMIZE_TASKS
    best_raw = _best_raw([_f(row.get("fitness")) for row in rows])
    ops = [0] * len(OPERATORS)
    outcomes = [0] * len(OUTCOMES)
    for row in rows:
        operator = row.get("operator") or ""
        if operator in OPERATORS:
            ops[OPERATORS.index(operator)] += 1
        outcome = row.get("outcome") or ""
        if outcome in OUTCOMES:
            outcomes[OUTCOMES.index(outcome)] += 1
    budget = _budget(config)
    finished = summary.get("status") == "finished"
    csv_path = run_dir / "evaluations.csv"
    last_ts = csv_path.stat().st_mtime if csv_path.is_file() else None
    stalled = (
        not finished and last_ts is not None and (time.time() - last_ts) > 30 * 60
    )
    match = RUN_NAME.match(run_dir.name)
    return {
        "task": task,
        "name": run_dir.name,
        "rep": int(match.group(1)) if match else 0,
        "backend": config.get("backend"),
        "seed": config.get("seed"),
        "budget": budget,
        "n_evals": len(rows),
        "finished": finished,
        "status": summary.get("status"),
        "stalled": stalled,
        "last_ts": last_ts,
        "created_at": config.get("created_at"),
        "minimize": minimize,
        "unit": TASK_UNIT.get(task, "fitness"),
        "label": TASK_LABEL.get(task, task),
        "best": None if best_raw is None else _score(best_raw, minimize),
        "curve": _best_curve(rows, minimize),
        "ops": ops,
        "outcomes": outcomes,
        "n_improve": outcomes[0],
        "critic_invalid": _critic_invalid(run_dir),
        "n_nodes": _n_nodes(run_dir, rows),
        "n_threads": _n_threads(run_dir),
    }


def run_payload(root: Path, task: str, name: str) -> dict[str, Any]:
    run_dir = _run_dir(root, task, name)
    config = _load_json(run_dir / "run_config.json")
    summary = _load_json(run_dir / "logs" / "summary.json")
    rows = slot_rows(_load_rows(run_dir / "evaluations.csv"))
    nodes = _load_nodes(run_dir)
    minimize = task in MINIMIZE_TASKS
    by_id = {node["id"]: node for node in nodes}
    layout = layout_formation_tree(
        [
            {
                "id": node["id"],
                "parent_id": node.get("parent_id") or 0,
                "created_slot": node.get("created_slot") or 0,
            }
            for node in nodes
        ]
    )
    best_id = _best_id(run_dir)
    return {
        "task": task,
        "name": name,
        "label": TASK_LABEL.get(task, task),
        "unit": TASK_UNIT.get(task, "fitness"),
        "minimize": minimize,
        "backend": config.get("backend"),
        "seed": config.get("seed"),
        "budget": _budget(config),
        "finished": summary.get("status") == "finished",
        "n_evals": len(rows),
        "n_nodes": len(nodes),
        "best_id": best_id,
        "best": None
        if best_id is None or by_id.get(best_id, {}).get("fitness") is None
        else _score(float(by_id[best_id]["fitness"]), minimize),
        "nodes": nodes,
        "layout": layout,
        "edges": _formation_edges(rows, by_id),
        "slots": [_slot_record(row, rows_full=None, minimize=minimize) for row in rows],
        "allocations": _allocations(run_dir),
        "threads": _threads(run_dir, minimize),
        "best_history": _best_history(run_dir, minimize),
        "curve": _best_curve(rows, minimize),
        "usage": _usage(summary),
    }


def node_payload(root: Path, task: str, name: str, node_id: int) -> dict[str, Any]:
    run_dir = _run_dir(root, task, name)
    algorithms = _checkpoint_nodes(run_dir)
    node = next((item for item in algorithms if item.get("id") == node_id), None)
    if node is None:
        raise KeyError(f"node {node_id} is not in {name}")
    minimize = task in MINIMIZE_TASKS
    path = _formation_nodes(algorithms, node_id)
    return {
        "id": node_id,
        "parent_id": node.get("parent_id"),
        "idea": node.get("idea"),
        "fitness": _f(node.get("fitness")),
        "score": None
        if node.get("fitness") is None
        else _score(float(node["fitness"]), minimize),
        "thread_id": node.get("thread_id"),
        "created_slot": node.get("slot"),
        "code": node.get("code") or "",
        "path": [
            {
                "id": item["id"],
                "parent_id": item.get("parent_id"),
                "idea": item.get("idea"),
                "fitness": _f(item.get("fitness")),
                "score": None
                if item.get("fitness") is None
                else _score(float(item["fitness"]), minimize),
                "thread_id": item.get("thread_id"),
                "created_slot": item.get("slot"),
            }
            for item in path
        ],
    }


def slot_payload(root: Path, task: str, name: str, slot: int) -> dict[str, Any]:
    run_dir = _run_dir(root, task, name)
    all_rows = _load_rows(run_dir / "evaluations.csv")
    rows = slot_rows(all_rows)
    row = next((item for item in rows if int(item["slot"]) == slot), None)
    if row is None:
        raise KeyError(f"slot {slot} is not in {name}")
    minimize = task in MINIMIZE_TASKS
    repairs = sum(1 for item in all_rows if item.get("slot") == str(slot)) - 1
    allocation = _allocations(run_dir, slots=(slot,))
    decision = _decision_at_slot(run_dir / "decisions.jsonl", slot)
    return {
        "row": _slot_record(row, rows_full=all_rows, minimize=minimize),
        "repairs": max(0, repairs),
        "allocation": allocation[0] if allocation else None,
        "decision": decision,
    }


def layout_formation_tree(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pack each root family left-to-right; depth increases downward."""
    children: dict[int, list[dict[str, Any]]] = {}
    for node in nodes:
        children.setdefault(int(node.get("parent_id") or 0), []).append(node)
    for family in children.values():
        family.sort(
            key=lambda item: (int(item.get("created_slot") or 0), int(item["id"]))
        )
    cursor = [0.0]

    def place(node: dict[str, Any], depth: int) -> None:
        kids = children.get(int(node["id"]), [])
        node["depth"] = depth
        node["y"] = float(depth)
        if not kids:
            node["x"] = cursor[0]
            cursor[0] += 1.0
            return
        for child in kids:
            place(child, depth + 1)
        node["x"] = 0.5 * (kids[0]["x"] + kids[-1]["x"])

    roots = children.get(0, [])
    for index, root in enumerate(roots):
        if index:
            cursor[0] += 0.8
        place(root, 0)
    return nodes


def slot_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Final evaluation row per primary slot (initial attempt plus repairs)."""
    by_slot: dict[int, dict[str, str]] = {}
    for row in rows:
        try:
            slot = int(row.get("slot") or 0)
        except ValueError:
            continue
        if slot <= 0:
            continue
        by_slot[slot] = row
    return [by_slot[slot] for slot in sorted(by_slot)]


def _run_dir(root: Path, task: str, name: str) -> Path:
    if task not in TASKS or not RUN_NAME.match(name) or "/" in name or "\\" in name:
        raise KeyError(f"unknown V10 run {task}/{name}")
    path = (root / task / "traceaad_v10" / name).resolve()
    expected = (root / task / "traceaad_v10").resolve()
    if path.parent != expected or not path.is_dir():
        raise KeyError(f"unknown V10 run {task}/{name}")
    return path


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _f(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _score(fitness: float, minimize: bool) -> float:
    return -fitness if minimize else fitness


def _budget(config: dict[str, Any]) -> int:
    params = config.get("method_params")
    if isinstance(params, dict) and isinstance(params.get("budget"), int):
        if params["budget"] > 0:
            return params["budget"]
    return 1000


def _best_raw(values: list[float | None]) -> float | None:
    known = [value for value in values if value is not None]
    return max(known) if known else None


def _best_curve(rows: list[dict[str, str]], minimize: bool) -> list[list[float]]:
    curve: list[list[float]] = []
    best: float | None = None
    for row in rows:
        fitness = _f(row.get("fitness"))
        if fitness is None:
            continue
        if best is None or fitness > best:
            best = fitness
            curve.append([int(row["slot"]), round(_score(fitness, minimize), 6)])
    if rows and curve and int(rows[-1]["slot"]) != curve[-1][0] and best is not None:
        curve.append([int(rows[-1]["slot"]), round(_score(best, minimize), 6)])
    return curve


def _n_nodes(run_dir: Path, rows: list[dict[str, str]]) -> int:
    view_state = _load_json(run_dir / "checkpoints" / "view.json")
    if isinstance(view_state.get("nodes"), list) and view_state["nodes"]:
        return len(view_state["nodes"])
    return sum(1 for row in rows if row.get("node_id"))


def _n_threads(run_dir: Path) -> int:
    view_state = _load_json(run_dir / "checkpoints" / "view.json")
    if isinstance(view_state.get("threads"), list) and view_state["threads"]:
        return len(view_state["threads"])
    seen = {
        int(item["id"])
        for item in _load_jsonl(run_dir / "threads.jsonl")
        if item.get("id") is not None
    }
    return len(seen)


def _critic_invalid(run_dir: Path) -> int:
    return sum(
        1
        for event in _load_jsonl(run_dir / "mechanism_events.jsonl")
        if event.get("event") == "allocation" and event.get("invalid")
    )


def _load_nodes(run_dir: Path) -> list[dict[str, Any]]:
    view_state = _load_json(run_dir / "checkpoints" / "view.json")
    if isinstance(view_state.get("nodes"), list) and view_state["nodes"]:
        return [_compact_node(item) for item in view_state["nodes"] if item.get("id")]
    return [
        _compact_node(item)
        for item in _checkpoint_nodes(run_dir)
        if item.get("id") not in {0, None}
    ]


def _compact_node(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(item["id"]),
        "parent_id": item.get("parent_id") or 0,
        "fitness": _f(item.get("fitness")),
        "thread_id": item.get("thread_id"),
        "idea": item.get("idea"),
        "created_slot": int(item.get("slot") or item.get("created_slot") or 0),
    }


def _checkpoint_nodes(run_dir: Path) -> list[dict[str, Any]]:
    payload = _load_json(run_dir / "checkpoints" / "latest.json")
    nodes = payload.get("nodes")
    return nodes if isinstance(nodes, list) else []


def _best_id(run_dir: Path) -> int | None:
    view_state = _load_json(run_dir / "checkpoints" / "view.json")
    if isinstance(view_state.get("best_node_id"), int):
        return view_state["best_node_id"]
    valid = [
        item for item in _load_nodes(run_dir) if item.get("fitness") is not None
    ]
    if not valid:
        return None
    return max(valid, key=lambda node: float(node["fitness"]))["id"]


def _slot_record(
    row: dict[str, str], rows_full: list[dict[str, str]] | None, minimize: bool
) -> dict[str, Any]:
    fitness = _f(row.get("fitness"))
    start_fitness = _f(row.get("start_fitness"))
    slot = int(row["slot"])
    if rows_full is None:
        attempts = 1
    else:
        attempts = sum(1 for item in rows_full if item.get("slot") == str(slot))
    return {
        "slot": slot,
        "operator": row.get("operator") or None,
        "start_id": int(row["start_id"]) if row.get("start_id") else None,
        "reference_id": int(row["reference_id"]) if row.get("reference_id") else None,
        "node_id": int(row["node_id"]) if row.get("node_id") else None,
        "created_thread": int(row["created_thread"])
        if row.get("created_thread")
        else None,
        "outcome": row.get("outcome") or None,
        "fitness": fitness,
        "score": None if fitness is None else _score(fitness, minimize),
        "start_fitness": start_fitness,
        "start_score": None
        if start_fitness is None
        else _score(start_fitness, minimize),
        "q_origin": _f(row.get("q_origin")),
        "attempt": attempts,
        "error": row.get("error") or None,
        "error_type": row.get("error_type") or None,
        "elapsed": _f(row.get("elapsed_seconds")),
    }


def _allocations(run_dir: Path, slots: tuple[int, ...] | None = None) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for event in _load_jsonl(run_dir / "mechanism_events.jsonl"):
        if event.get("event") != "allocation":
            continue
        slot = int(event.get("slot") or 0)
        if slots is not None and slot not in slots:
            continue
        chosen = event.get("chosen") or {}
        selected.append(
            {
                "slot": slot,
                "round": event.get("round"),
                "invalid": bool(event.get("invalid")),
                "chosen": {
                    "opportunity_id": chosen.get("opportunity_id"),
                    "operator": chosen.get("operator"),
                    "start_id": chosen.get("start_id"),
                    "reference_id": chosen.get("reference_id"),
                    "rank": chosen.get("rank"),
                    "coverage": chosen.get("coverage"),
                },
                "competitive": [
                    {
                        "opportunity_id": entry.get("opportunity_id"),
                        "operator": entry.get("operator"),
                        "start_id": entry.get("start_id"),
                        "rank": entry.get("rank"),
                    }
                    for entry in event.get("competitive_set") or []
                ],
                "not_applicable": event.get("not_applicable") or [],
            }
        )
    return selected


def _threads(run_dir: Path, minimize: bool) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for item in _load_jsonl(run_dir / "threads.jsonl"):
        if item.get("id") is None:
            continue
        latest[int(item["id"])] = item
    threads = []
    for thread_id, item in sorted(latest.items()):
        best_history = [_f(value) for value in item.get("best_history") or []]
        known = [value for value in best_history if value is not None]
        entry: dict[str, Any] = {
            "id": thread_id,
            "origin_action": item.get("origin_action"),
            "origin_idea": item.get("origin_idea"),
            "origin_slot": item.get("origin_slot"),
            "created_node_id": item.get("created_node_id"),
            "q_origin": _f(item.get("q_origin")),
            "opportunities_used": item.get("opportunities_used"),
            "best_score": None
            if not known
            else _score(known[-1], minimize),
            "history_tail": [
                round(_score(value, minimize), 6)
                for value in known[-THREAD_HISTORY_KEEP:]
            ],
        }
        for horizon in G_HORIZONS:
            entry[f"G{horizon}"] = _f(item.get(f"G{horizon}"))
        threads.append(entry)
    return threads


def _best_history(run_dir: Path, minimize: bool) -> list[dict[str, Any]]:
    history = []
    for item in _load_jsonl(run_dir / "best_history.jsonl"):
        fitness = _f(item.get("fitness"))
        if fitness is None:
            continue
        history.append(
            {
                "slot": int(item.get("slot") or 0),
                "fitness": fitness,
                "score": _score(fitness, minimize),
                "node_id": item.get("node_id"),
                "idea": item.get("idea"),
                "thread_id": item.get("thread_id"),
            }
        )
    return history


def _formation_edges(
    rows: list[dict[str, str]], by_id: dict[int, dict[str, Any]]
) -> list[dict[str, Any]]:
    row_by_child = {
        int(row["node_id"]): row for row in rows if row.get("node_id")
    }
    edges = []
    for node in by_id.values():
        parent = int(node.get("parent_id") or 0)
        if parent <= 0:
            continue
        row = row_by_child.get(node["id"], {})
        edges.append(
            {
                "source": parent,
                "target": node["id"],
                "operator": row.get("operator"),
                "outcome": row.get("outcome"),
            }
        )
    return edges


def _formation_nodes(algorithms: list[dict[str, Any]], node_id: int) -> list[dict[str, Any]]:
    by_id = {int(item["id"]): item for item in algorithms if item.get("id") is not None}
    path: list[dict[str, Any]] = []
    current = node_id
    seen: set[int] = set()
    while current and current not in seen:
        seen.add(current)
        node = by_id.get(current)
        if node is None or current == 0:
            break
        path.append(node)
        current = int(node.get("parent_id") or 0)
    path.reverse()
    return path


def _decision_at_slot(path: Path, slot: int) -> dict[str, Any] | None:
    """Critic + generation decisions at one slot, without prompt/response bodies."""
    critic = None
    generation = None
    for item in _load_jsonl(path):
        if int(item.get("slot") or -1) != slot:
            continue
        if item.get("stage") == "critic" and critic is None:
            critic = {
                "prompt_chars": item.get("prompt_chars"),
                "prompt_char_budget": item.get("prompt_char_budget"),
                "prompt_code_clipped": item.get("prompt_code_clipped"),
                "dropped_reference_codes": item.get("dropped_reference_codes"),
                "invalid": item.get("invalid"),
                "competitive_set": [
                    {
                        "opportunity_id": entry.get("opportunity_id"),
                        "operator": entry.get("operator"),
                        "start_id": entry.get("start_id"),
                        "reference_id": entry.get("reference_id"),
                        "rank": entry.get("rank"),
                        "reason": entry.get("reason"),
                        "evidence_refs": entry.get("evidence_refs"),
                        "expected_payoff_horizon": entry.get("expected_payoff_horizon"),
                        "semantic_mismatch": entry.get("semantic_mismatch"),
                    }
                    for entry in item.get("competitive_set") or []
                ],
                "not_applicable": item.get("not_applicable"),
            }
        if item.get("stage") == "generation" and generation is None:
            generation = {
                "opportunity_id": item.get("opportunity_id"),
                "operator": item.get("operator"),
                "start_id": item.get("start_id"),
                "reference_id": item.get("reference_id"),
                "critic_rank": item.get("critic_rank"),
                "idea": item.get("idea"),
            }
    if critic is None and generation is None:
        return None
    return {"critic": critic, "generation": generation}


def _usage(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "generation_llm_calls",
        "critic_llm_calls",
        "critic_invalid",
        "repair_llm_calls",
        "evaluator_call_count",
        "budget_slots",
    )
    token_keys = (
        "generation_prompt_tokens",
        "generation_completion_tokens",
        "critic_prompt_tokens",
        "critic_completion_tokens",
        "repair_prompt_tokens",
        "repair_completion_tokens",
    )
    usage = {key: summary.get(key) for key in keys if summary.get(key) is not None}
    usage.update(
        {key: summary.get(key) for key in token_keys if summary.get(key) is not None}
    )
    return usage

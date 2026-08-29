"""Read V9.19 run artifacts into the live observer payloads."""

from __future__ import annotations

import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from llm4ad.method.traceaad_v9_19.schema import (
    MIN_NEIGHBORS,
    NEIGHBORHOOD_FRACTION,
)

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
RUN_NAME = re.compile(r"^v9_19_.+_rep(\d+)$")
ACTIONS = ("develop", "explore")
OUTCOMES = (
    "improve",
    "plateau",
    "regress",
    "invalid",
    "duplicate",
    "timeout",
    "profiling_failed",
)


def discover_runs(root: Path) -> list[Path]:
    runs = [
        path
        for path in sorted(root.glob("*/traceaad_v9_19/v9_19_*"))
        if path.is_dir()
        and "smoke" not in path.name
        and (path / "evaluations.csv").is_file()
        and RUN_NAME.match(path.name)
    ]
    return runs


def overview_payload(root: Path) -> dict[str, Any]:
    return {
        "budget": 1000,
        "runs": [overview_run(path) for path in discover_runs(root)],
    }


def overview_run(run_dir: Path) -> dict[str, Any]:
    config = _load_json(run_dir / "run_config.json")
    summary = _load_json(run_dir / "logs" / "summary.json")
    rows = settle_rows(_load_rows(run_dir / "evaluations.csv"))
    task = str(config.get("task") or run_dir.parent.parent.name)
    minimize = task in MINIMIZE_TASKS
    n_evals = max((int(row["slot"]) for row in rows), default=0)
    fitnesses = [_f(row.get("fitness")) for row in rows]
    best_raw = _best_raw(fitnesses)
    curve = _best_curve(rows, minimize)
    ops = [0, 0]
    outcomes = [0] * len(OUTCOMES)
    for row in rows:
        action = row.get("action") or ""
        if action in ACTIONS:
            ops[ACTIONS.index(action)] += 1
        outcome = row.get("outcome") or ""
        if outcome in OUTCOMES:
            outcomes[OUTCOMES.index(outcome)] += 1
    budget = 1000
    params = config.get("method_params") if isinstance(config.get("method_params"), dict) else {}
    if isinstance(params.get("budget"), int) and params["budget"] > 0:
        budget = params["budget"]
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
        "n_evals": n_evals,
        "finished": finished,
        "stalled": stalled,
        "last_ts": last_ts,
        "created_at": config.get("created_at"),
        "minimize": minimize,
        "unit": TASK_UNIT.get(task, "fitness"),
        "label": TASK_LABEL.get(task, task),
        "best": None if best_raw is None else _score(best_raw, minimize),
        "curve": curve,
        "ops": ops,
        "outcomes": outcomes,
        "n_improve": outcomes[0],
        "n_nodes": _n_nodes(run_dir, rows),
    }


def run_payload(root: Path, task: str, name: str) -> dict[str, Any]:
    run_dir = _run_dir(root, task, name)
    config = _load_json(run_dir / "run_config.json")
    summary = _load_json(run_dir / "logs" / "summary.json")
    rows = settle_rows(_load_rows(run_dir / "evaluations.csv"))
    nodes = _load_nodes(run_dir)
    minimize = task in MINIMIZE_TASKS
    by_id = {node["id"]: node for node in nodes}
    slots = [_slot_record(row, minimize) for row in rows]
    selected = _selected_parents(run_dir, rows)
    landscape = _landscape(run_dir, by_id, minimize)
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
    best_id = _best_id(run_dir, nodes)
    n_evals = max((int(row["slot"]) for row in rows), default=0)
    return {
        "task": task,
        "name": name,
        "label": TASK_LABEL.get(task, task),
        "unit": TASK_UNIT.get(task, "fitness"),
        "minimize": minimize,
        "backend": config.get("backend"),
        "seed": config.get("seed"),
        "budget": (config.get("method_params") or {}).get("budget", 1000)
        if isinstance(config.get("method_params"), dict)
        else 1000,
        "finished": summary.get("status") == "finished",
        "n_evals": n_evals,
        "n_nodes": len(nodes),
        "best_id": best_id,
        "best": None
        if best_id is None or by_id.get(best_id, {}).get("fitness") is None
        else _score(float(by_id[best_id]["fitness"]), minimize),
        "nodes": nodes,
        "layout": layout,
        "edges": _formation_edges(nodes, rows),
        "slots": slots,
        "selected": selected,
        "best_history": _best_history(run_dir, rows, by_id, minimize),
        "landscape": landscape,
        "curve": _best_curve(rows, minimize),
    }


def node_payload(root: Path, task: str, name: str, node_id: int) -> dict[str, Any]:
    run_dir = _run_dir(root, task, name)
    algorithms = _checkpoint_algorithms(run_dir)
    node = next((item for item in algorithms if item.get("id") == node_id), None)
    if node is None:
        raise KeyError(f"node {node_id} is not in {name}")
    minimize = task in MINIMIZE_TASKS
    path = _formation_nodes(algorithms, node_id)
    return {
        "id": node_id,
        "parent_id": node.get("parent_id"),
        "idea": node.get("idea"),
        "action": node.get("action"),
        "fitness": node.get("fitness"),
        "score": None
        if node.get("fitness") is None
        else _score(float(node["fitness"]), minimize),
        "t_response": node.get("t_response"),
        "novelty": node.get("novelty"),
        "behavior": node.get("behavior_tag"),
        "opportunities": node.get("opportunities"),
        "created_slot": node.get("created_slot"),
        "code": node.get("code") or "",
        "path": [
            {
                "id": item["id"],
                "parent_id": item.get("parent_id"),
                "idea": item.get("idea"),
                "action": item.get("action"),
                "fitness": item.get("fitness"),
                "score": None
                if item.get("fitness") is None
                else _score(float(item["fitness"]), minimize),
                "t_response": item.get("t_response"),
                "novelty": item.get("novelty"),
                "behavior": item.get("behavior_tag"),
                "created_slot": item.get("created_slot"),
            }
            for item in path
        ],
    }


def slot_payload(root: Path, task: str, name: str, slot: int) -> dict[str, Any]:
    run_dir = _run_dir(root, task, name)
    rows = settle_rows(_load_rows(run_dir / "evaluations.csv"))
    row = next((item for item in rows if int(item["slot"]) == slot), None)
    if row is None:
        raise KeyError(f"slot {slot} is not in {name}")
    minimize = task in MINIMIZE_TASKS
    events = _load_jsonl(run_dir / "mechanism_events.jsonl")
    pre_event, action_event = _decision_events_for_slot(events, rows, slot)
    ranking = sorted(
        (pre_event or {}).get("snapshot") or [],
        key=lambda item: (-float(item.get("S") or 0), int(item.get("id") or 0)),
    )
    decision = _decision_at_slot(run_dir / "decisions.jsonl", slot)
    if decision is not None:
        decision = {
            key: value
            for key, value in decision.items()
            if key not in {"exact_prompt", "exact_response", "current_code"}
        }
    return {
        "row": _slot_record(row, minimize),
        "ranking": ranking,
        "action": action_event,
        "decision": decision,
    }


def layout_formation_tree(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pack each root family left-to-right; depth increases downward."""
    children: dict[int, list[dict[str, Any]]] = {}
    for node in nodes:
        children.setdefault(int(node.get("parent_id") or 0), []).append(node)
    for family in children.values():
        family.sort(key=lambda item: (int(item.get("created_slot") or 0), int(item["id"])))
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


def classical_mds(matrix: np.ndarray) -> np.ndarray:
    squared = matrix ** 2
    size = len(matrix)
    centerer = np.eye(size) - np.ones((size, size)) / size
    centered = -0.5 * centerer @ squared @ centerer
    centered = (centered + centered.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order][:2], 0.0, None)
    return eigenvectors[:, order[:2]] * np.sqrt(eigenvalues)


def settle_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
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
        raise KeyError(f"unknown V9.19 run {task}/{name}")
    path = (root / task / "traceaad_v9_19" / name).resolve()
    expected = (root / task / "traceaad_v9_19").resolve()
    if path.parent != expected or not path.is_dir():
        raise KeyError(f"unknown V9.19 run {task}/{name}")
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
    return sum(1 for row in rows if row.get("child_id"))


def _load_nodes(run_dir: Path) -> list[dict[str, Any]]:
    view_state = _load_json(run_dir / "checkpoints" / "view.json")
    if isinstance(view_state.get("nodes"), list) and view_state["nodes"]:
        return [_compact_node(item) for item in view_state["nodes"] if item.get("id")]
    return [
        _compact_node(item)
        for item in _checkpoint_algorithms(run_dir)
        if item.get("id") not in {0, None}
    ]


def _compact_node(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": int(item["id"]),
        "parent_id": item.get("parent_id") or 0,
        "fitness": _f(item.get("fitness")),
        "idea": item.get("idea"),
        "action": item.get("action"),
        "created_slot": int(item.get("created_slot") or 0),
        "t_response": _f(item.get("t_response")),
        "novelty": _f(item.get("novelty")),
        "behavior": item.get("behavior_tag") or item.get("behavior"),
        "opportunities": int(item.get("opportunities") or 0),
    }


def _checkpoint_algorithms(run_dir: Path) -> list[dict[str, Any]]:
    payload = _load_json(run_dir / "checkpoints" / "latest.json")
    tree = payload.get("tree") if isinstance(payload.get("tree"), dict) else {}
    algorithms = tree.get("algorithms")
    return algorithms if isinstance(algorithms, list) else []


def _best_id(run_dir: Path, nodes: list[dict[str, Any]]) -> int | None:
    view_state = _load_json(run_dir / "checkpoints" / "view.json")
    if isinstance(view_state.get("best_id"), int):
        return view_state["best_id"]
    valid = [node for node in nodes if node.get("fitness") is not None]
    if not valid:
        return None
    return max(valid, key=lambda node: float(node["fitness"]))["id"]


def _slot_record(row: dict[str, str], minimize: bool) -> dict[str, Any]:
    fitness = _f(row.get("fitness"))
    parent_fitness = _f(row.get("parent_fitness"))
    return {
        "slot": int(row["slot"]),
        "parent_id": int(row["parent_id"] or 0),
        "child_id": int(row["child_id"]) if row.get("child_id") else None,
        "action": row.get("action") or None,
        "mode": row.get("mode") or None,
        "outcome": row.get("outcome") or None,
        "status": row.get("status") or None,
        "fitness": fitness,
        "score": None if fitness is None else _score(fitness, minimize),
        "parent_fitness": parent_fitness,
        "parent_score": None if parent_fitness is None else _score(parent_fitness, minimize),
        "p_explore": _f(row.get("p_explore")),
        "t_response": _f(row.get("t_response")),
        "beta": _f(row.get("beta")),
        "ess": _f(row.get("ess")),
        "pool_size": int(row["pool_size"]) if row.get("pool_size") else None,
        "neighborhood_size": int(row["neighborhood_size"])
        if row.get("neighborhood_size")
        else None,
        "attempt": int(row["attempt"]) if row.get("attempt") else 1,
        "error": row.get("error") or None,
        "error_type": row.get("error_type") or None,
        "elapsed": _f(row.get("elapsed_seconds")),
    }


def _ordinary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("action")]


def _decision_events_for_slot(
    events: list[dict[str, Any]], rows: list[dict[str, str]], slot: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    pres = [event for event in events if event.get("event") == "pre_decision"]
    actions = [event for event in events if event.get("event") == "action_decision"]
    ordinary = _ordinary_rows(rows)
    index = next((i for i, row in enumerate(ordinary) if int(row["slot"]) == slot), None)
    if index is not None and index < len(pres):
        return pres[index], actions[index] if index < len(actions) else None
    pre = next((event for event in pres if int(event.get("slot") or -1) in {slot, slot - 1}), None)
    action = next(
        (event for event in actions if int(event.get("slot") or -1) in {slot, slot - 1}),
        None,
    )
    return pre, action


def _selected_parents(run_dir: Path, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    events = _load_jsonl(run_dir / "mechanism_events.jsonl")
    pres = [event for event in events if event.get("event") == "pre_decision"]
    actions = [event for event in events if event.get("event") == "action_decision"]
    selected: list[dict[str, Any]] = []
    for index, row in enumerate(_ordinary_rows(rows)):
        action_event = actions[index] if index < len(actions) else {}
        pre_event = pres[index] if index < len(pres) else {}
        parent_id = int(
            action_event.get("parent_id") or pre_event.get("parent_id") or row.get("parent_id") or 0
        )
        snapshot = {
            int(entry["id"]): entry for entry in pre_event.get("snapshot") or []
        }
        entry = snapshot.get(parent_id, {})
        selected.append(
            {
                "slot": int(row["slot"]),
                "decision_index": action_event.get("decision_index", pre_event.get("decision_index")),
                "parent_id": parent_id,
                "action": action_event.get("action") or row.get("action"),
                "P": _f(entry.get("P")),
                "U": _f(entry.get("U")),
                "T": _f(
                    action_event.get("T")
                    if action_event.get("T") is not None
                    else entry.get("T")
                ),
                "S": _f(entry.get("S")),
                "q": _f(entry.get("q")),
                "p_explore": _f(action_event.get("p_explore") or row.get("p_explore")),
                "beta": _f(row.get("beta")),
                "ess": _f(row.get("ess")),
                "outcome": row.get("outcome"),
            }
        )
    return selected


def _formation_edges(
    nodes: list[dict[str, Any]], rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    outcome_by_child = {
        int(row["child_id"]): row.get("outcome")
        for row in rows
        if row.get("child_id")
    }
    edges = []
    for node in nodes:
        parent = int(node.get("parent_id") or 0)
        if parent <= 0:
            continue
        edges.append(
            {
                "source": parent,
                "target": node["id"],
                "action": node.get("action"),
                "outcome": outcome_by_child.get(node["id"]),
            }
        )
    return edges


def _best_history(
    run_dir: Path,
    rows: list[dict[str, str]],
    by_id: dict[int, dict[str, Any]],
    minimize: bool,
) -> list[dict[str, Any]]:
    reconstructed = _reconstruct_best_history(rows, by_id, minimize)
    recorded = []
    for item in _load_jsonl(run_dir / "best_history.jsonl"):
        fitness = _f(item.get("fitness"))
        if fitness is None:
            continue
        recorded.append(
            {
                "slot": int(item.get("slot") or item.get("eval_count") or 0),
                "fitness": fitness,
                "child_id": int(item["child_id"]) if item.get("child_id") else None,
                "idea": item.get("idea"),
                "action": item.get("action"),
                "novelty": _f(item.get("novelty")),
                "behavior": item.get("behavior"),
                "t_response": _f(item.get("t_response")),
                "score": _score(fitness, minimize),
            }
        )
    if not recorded:
        return reconstructed
    start = recorded[0]["slot"]
    prefix = [item for item in reconstructed if item["slot"] < start]
    return prefix + recorded


def _reconstruct_best_history(
    rows: list[dict[str, str]],
    by_id: dict[int, dict[str, Any]],
    minimize: bool,
) -> list[dict[str, Any]]:
    history = []
    best: float | None = None
    for row in rows:
        fitness = _f(row.get("fitness"))
        child_id = int(row["child_id"]) if row.get("child_id") else None
        if fitness is None or child_id is None:
            continue
        if best is not None and fitness <= best:
            continue
        best = fitness
        node = by_id.get(child_id, {})
        history.append(
            {
                "slot": int(row["slot"]),
                "fitness": fitness,
                "child_id": child_id,
                "idea": node.get("idea"),
                "action": node.get("action"),
                "novelty": node.get("novelty"),
                "behavior": node.get("behavior"),
                "t_response": node.get("t_response"),
                "score": _score(fitness, minimize),
            }
        )
    return history


def _landscape(
    run_dir: Path, by_id: dict[int, dict[str, Any]], minimize: bool
) -> dict[str, Any] | None:
    path = run_dir / "checkpoints" / "behave.npz"
    if not path.is_file():
        return None
    with np.load(path, mmap_mode="r") as payload:
        if "matrix" not in payload.files or "ids" not in payload.files:
            return None
        matrix = np.asarray(payload["matrix"], dtype=np.float64)
        ids = [int(value) for value in np.asarray(payload["ids"]).tolist()]
    if len(ids) < 2 or matrix.shape != (len(ids), len(ids)):
        return None
    coordinates = classical_mds(matrix)
    xy = [
        [round(float(coordinates[index, 0]), 5), round(float(coordinates[index, 1]), 5)]
        for index in range(len(ids))
    ]
    return {
        "ids": ids,
        "xy": xy,
        "knn": _knn_edges(ids, matrix),
        "scores": [
            None
            if by_id.get(node_id, {}).get("fitness") is None
            else _score(float(by_id[node_id]["fitness"]), minimize)
            for node_id in ids
        ],
    }


def _knn_edges(ids: list[int], matrix: np.ndarray) -> list[list[int]]:
    size = len(ids)
    if size < 2:
        return []
    k = min(size - 1, max(MIN_NEIGHBORS, math.ceil(NEIGHBORHOOD_FRACTION * (size - 1))))
    filled = matrix.copy()
    np.fill_diagonal(filled, np.inf)
    edges: list[list[int]] = []
    for index in range(size):
        order = np.argsort(filled[index], kind="stable")
        for neighbor in order[:k]:
            edges.append([ids[index], ids[int(neighbor)]])
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
    for item in _load_jsonl(path):
        if int(item.get("slot") or -1) == slot:
            return item
    return None

"""V9.19 search-observer payloads and HTTP surface."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.plotting import v919_view as view
from llm4ad.method.traceaad_v9_19.artifacts import RunArtifacts

OUTCOME_HEADER = [
    "slot", "parent_id", "child_id", "action", "mode",
    "outcome", "status", "fitness", "parent_fitness", "error",
    "p_explore", "t_response", "beta", "ess", "pool_size", "neighborhood_size",
    "attempt", "attempt_kind", "request_seed", "elapsed_seconds", "error_type",
]


def _node(node_id: int, parent: int | None, *, slot: int, fitness: float, action: str | None) -> dict:
    return {
        "id": node_id,
        "code": f"def select_{node_id}():\n    return {node_id}\n",
        "fitness": fitness,
        "parent_id": parent,
        "idea": None if action is None else f"idea-{node_id}",
        "action": action,
        "created_slot": slot,
        "t_response": 0.5 if action is None else 0.4 + 0.02 * node_id,
        "novelty": None if action is None else 0.2 * (node_id % 5),
        "behavior_tag": None if action is None else (
            "near-known" if node_id % 3 == 0 else "intermediate" if node_id % 3 == 1 else "far-from-archive"
        ),
        "opportunities": node_id % 4,
    }


def write_v919_run(
    root: Path,
    *,
    task: str = "tsp_construct",
    name: str = "v9_19_tsp_construct_rep1",
    n_roots: int = 8,
    n_ordinary: int = 6,
    with_view: bool = False,
    with_best_history: bool = False,
) -> Path:
    run_dir = root / task / "traceaad_v9_19" / name
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "logs").mkdir(parents=True)
    n_nodes = n_roots + n_ordinary

    algorithms = [
        {
            "id": 0,
            "code": None,
            "fitness": None,
            "parent_id": None,
            "idea": None,
            "action": None,
            "created_slot": 0,
            "t_response": 0.5,
            "novelty": None,
            "behavior_tag": None,
            "opportunities": 0,
        }
    ]
    rows: list[list[object]] = []
    events: list[dict] = []
    for node in range(1, n_nodes + 1):
        slot = node
        parent = 0 if node <= n_roots else ((node - n_roots - 1) % n_roots) + 1
        action = None if node <= n_roots else ("develop" if node % 2 == 0 else "explore")
        fitness = -10.0 + 0.25 * node
        outcome = "plateau" if node <= n_roots else ("improve" if node % 3 == 0 else "regress")
        algorithms.append(
            _node(node, parent if parent else 0, slot=slot, fitness=fitness, action=action)
        )
        rows.append(
            [
                slot, parent, node, action or "",
                "initialization" if action is None else "ordinary",
                outcome, "ok", fitness, None if parent == 0 else -10.0 + 0.25 * parent, "",
                "" if action is None else 0.35, "" if action is None else 0.45,
                "" if action is None else 1.2, "" if action is None else 2.0,
                "" if action is None else n_roots + (node - n_roots - 1),
                "" if action is None else 2, 1, "initial", slot, 0.4, "",
            ]
        )
        events.append(
            {
                "event": "new_node",
                "slot": slot,
                "node_id": node,
                "parent_id": parent,
                "action": action,
                "fitness": fitness,
                "outcome": outcome,
                "T": algorithms[-1]["t_response"],
                "novelty": algorithms[-1]["novelty"],
                "behavior": algorithms[-1]["behavior_tag"],
            }
        )

    # One repaired slot: first attempt invalid, second settles.
    rows.append(
        [
            n_nodes, 1, "", "develop", "ordinary", "invalid", "invalid_result",
            "", -10.0, "boom", 0.4, 0.5, 1.0, 2.0, n_roots, 2, 1, "initial",
            99, 0.1, "InvalidEvaluationResult",
        ]
    )
    rows.append(
        [
            n_nodes, 1, n_nodes, "develop", "ordinary", "regress", "ok",
            -10.0 + 0.25 * n_nodes, -10.0, "", 0.4, 0.5, 1.0, 2.0, n_roots, 2,
            2, "repair", 99, 0.2, "",
        ]
    )

    for decision in range(n_ordinary):
        parent = (decision % n_roots) + 1
        snapshot = [
            {
                "id": node,
                "q": 0.1 * (node % 5),
                "P": 0.4 + 0.01 * node,
                "U": 0.6 - 0.01 * node,
                "B": float(node % 3),
                "c_t": node % 4,
                "T": 0.5,
                "S": 0.35 + 0.01 * node,
            }
            for node in range(1, n_roots + decision + 1)
        ]
        events.append(
            {
                "event": "pre_decision",
                "decision_index": decision,
                "slot": n_roots + decision,
                "parent_id": parent,
                "pool_size": len(snapshot),
                "neighborhood_size": 2,
                "beta": 1.5 + decision,
                "ess": 2.0,
                "marker": 0.4,
                "snapshot": snapshot,
            }
        )
        events.append(
            {
                "event": "action_decision",
                "decision_index": decision,
                "slot": n_roots + decision,
                "parent_id": parent,
                "T": 0.42,
                "p_explore": 0.33,
                "action": "develop" if decision % 2 == 0 else "explore",
                "action_draw": 0.2,
            }
        )

    with (run_dir / "evaluations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTCOME_HEADER)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])

    with (run_dir / "mechanism_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")
        handle.write('{"event": "pre_deci')

    with (run_dir / "decisions.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "task": task,
                    "slot": n_roots + 1,
                    "decision_index": 0,
                    "parent_id": 1,
                    "current_code": "parent-code",
                    "formation_path": [],
                    "action": "develop",
                    "llm_output": {"idea": "idea-child", "code": "child-code"},
                    "q_p": -9.75,
                    "q_c": -9.5,
                    "result": "improve",
                    "nu": 0.4,
                    "behavior": "intermediate",
                    "P": 0.5,
                    "U": 0.5,
                    "T": 0.5,
                    "exact_prompt": "PROMPT",
                    "exact_response": "RESPONSE",
                }
            )
            + "\n"
        )

    n = n_nodes
    distances = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(np.float32)
    np.savez(
        run_dir / "checkpoints" / "behave.npz",
        ids=np.arange(1, n + 1, dtype=np.int64),
        matrix=distances,
        states=np.zeros((n, 1, 1, 1), dtype=np.int32),
        lengths=np.ones((n, 1, 1), dtype=np.int32),
    )
    (run_dir / "checkpoints" / "latest.json").write_text(
        json.dumps(
            {
                "version": "v9_19",
                "task": task,
                "tree": {"algorithms": algorithms},
                "n_eval": n_nodes,
            }
        ),
        encoding="utf-8",
    )
    if with_view:
        compact = [{k: v for k, v in item.items() if k != "code"} for item in algorithms[1:]]
        (run_dir / "checkpoints" / "view.json").write_text(
            json.dumps({"n_eval": n_nodes, "best_id": n_nodes, "nodes": compact}),
            encoding="utf-8",
        )
    if with_best_history:
        (run_dir / "best_history.jsonl").write_text(
            json.dumps(
                {
                    "slot": 3,
                    "fitness": -9.25,
                    "child_id": 3,
                    "idea": "first-best",
                    "action": None,
                    "program": "def select_3():\n    return 3\n",
                }
            )
            + "\n",
            encoding="utf-8",
        )
    (run_dir / "best_program.py").write_text("# Fitness: -6.5\n\ndef select():\n    return 1\n")
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "backend": "local",
                "created_at": "2026-08-28T16:38:00",
                "method": "traceaad_v9_19",
                "method_params": {"budget": 1000},
                "repeat": 1,
                "run_name": name,
                "seed": 0,
                "task": task,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "logs" / "summary.json").write_text("{}", encoding="utf-8")
    return run_dir


def test_discover_only_v919_run_directories(tmp_path: Path) -> None:
    write_v919_run(tmp_path)
    (tmp_path / "tsp_construct" / "traceaad_v9_16" / "v9_16_1").mkdir(parents=True)
    found = view.discover_runs(tmp_path)
    assert [path.name for path in found] == ["v9_19_tsp_construct_rep1"]


def test_overview_reports_best_curve_and_action_counts(tmp_path: Path) -> None:
    write_v919_run(tmp_path)
    payload = view.overview_payload(tmp_path)
    assert payload["budget"] == 1000
    assert len(payload["runs"]) == 1
    run = payload["runs"][0]
    assert run["task"] == "tsp_construct"
    assert run["rep"] == 1
    assert run["n_evals"] == 14
    assert run["minimize"] is True
    assert run["ops"] == [3, 3]
    assert run["best"] == pytest.approx(10.0 - 0.25 * 14)
    assert run["curve"][-1][1] == pytest.approx(run["best"])
    assert run["backend"] == "local"


def test_run_payload_has_tree_landscape_and_settled_slots(tmp_path: Path) -> None:
    write_v919_run(tmp_path)
    payload = view.run_payload(tmp_path, "tsp_construct", "v9_19_tsp_construct_rep1")
    assert payload["n_evals"] == 14
    assert len(payload["nodes"]) == 14
    assert "code" not in payload["nodes"][0]
    assert payload["nodes"][0]["idea"] is None
    assert payload["nodes"][-1]["idea"] == "idea-14"
    assert len(payload["slots"]) == 14
    last = payload["slots"][-1]
    assert last["slot"] == 14
    assert last["outcome"] == "regress"
    assert last["attempt"] == 2
    assert len(payload["layout"]) == 14
    assert payload["landscape"] is not None
    assert len(payload["landscape"]["ids"]) == 14
    assert len(payload["landscape"]["xy"]) == 14
    assert payload["landscape"]["knn"]
    assert payload["selected"][0]["parent_id"] == 1
    assert set(payload["selected"][0]) >= {"P", "U", "T", "S", "action", "p_explore"}
    assert payload["best_history"][-1]["child_id"] == 14
    assert "program" not in payload["best_history"][0]


def test_view_json_is_used_when_present(tmp_path: Path) -> None:
    write_v919_run(tmp_path, with_view=True)
    payload = view.run_payload(tmp_path, "tsp_construct", "v9_19_tsp_construct_rep1")
    assert payload["best_id"] == 14
    assert all("code" not in node for node in payload["nodes"])


def test_best_history_file_wins_over_reconstruction(tmp_path: Path) -> None:
    write_v919_run(tmp_path, with_best_history=True)
    payload = view.run_payload(tmp_path, "tsp_construct", "v9_19_tsp_construct_rep1")
    by_slot = {item["slot"]: item for item in payload["best_history"]}
    assert by_slot[3]["idea"] == "first-best"
    assert by_slot[3]["score"] == pytest.approx(9.25)
    assert 1 in by_slot and 2 in by_slot
    assert payload["best_history"][-1]["slot"] == 3


def test_node_and_slot_endpoints(tmp_path: Path) -> None:
    write_v919_run(tmp_path)
    node = view.node_payload(tmp_path, "tsp_construct", "v9_19_tsp_construct_rep1", 10)
    assert "return 10" in node["code"]
    assert node["idea"] == "idea-10"
    assert node["path"][-1]["id"] == 10
    assert node["parent_id"] in {step["id"] for step in node["path"][:-1]}
    slot = view.slot_payload(tmp_path, "tsp_construct", "v9_19_tsp_construct_rep1", 9)
    assert slot["row"]["action"] == "explore"
    assert slot["ranking"][0]["id"]
    assert slot["decision"]["llm_output"]["idea"] == "idea-child"
    assert "exact_prompt" not in slot["decision"]


def test_record_best_writes_history_jsonl(tmp_path: Path) -> None:
    artifacts = RunArtifacts(tmp_path, console_output=False)
    artifacts.record_best(
        code="def f():\n    return 1\n",
        fitness=-6.5,
        slot=4,
        child_id=4,
        idea="cut corners",
        action="develop",
        novelty=0.2,
        behavior="near-known",
        t_response=0.7,
    )
    artifacts.finish()
    lines = (tmp_path / "best_history.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    assert record["slot"] == 4
    assert record["child_id"] == 4
    assert record["idea"] == "cut corners"
    assert "def f()" in record["program"]
    text = (tmp_path / "best_program.py").read_text(encoding="utf-8")
    assert "Fitness: -6.5" in text


def test_tree_layout_places_roots_and_children() -> None:
    nodes = [
        {"id": 1, "parent_id": 0, "created_slot": 1},
        {"id": 2, "parent_id": 0, "created_slot": 2},
        {"id": 3, "parent_id": 1, "created_slot": 9},
    ]
    layout = {item["id"]: item for item in view.layout_formation_tree(nodes)}
    assert layout[1]["depth"] == 0
    assert layout[3]["depth"] == 1
    assert layout[3]["y"] > layout[1]["y"]
    assert layout[1]["x"] != layout[2]["x"]

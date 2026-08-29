"""Tests for the V9.19 training-process visualization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from experiments.plotting.plot_v919_run import (  # noqa: E402
    RunData,
    classical_mds,
    neighborhood_edges,
    plot_dashboard,
    plot_landscape,
    plot_task_curves,
    rolling_fractions,
)

OUTCOME_HEADER = [
    "slot", "parent_id", "child_id", "action", "mode",
    "outcome", "status", "fitness", "parent_fitness", "error",
    "p_explore", "t_response", "beta", "ess", "pool_size", "neighborhood_size",
    "attempt", "attempt_kind", "request_seed", "elapsed_seconds", "error_type",
]


def _write_run(root: Path, *, n_roots: int = 8, n_ordinary: int = 6) -> Path:
    """Build a minimal but complete V9.19 run artifact set."""
    run_dir = root / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    n_nodes = n_roots + n_ordinary

    algorithms = [{"id": 0, "parent_id": None}]
    rows: list[list[object]] = []
    for node in range(1, n_nodes + 1):
        slot = node
        parent = 0 if node <= n_roots else node - n_roots
        action = "" if node <= n_roots else (
            "develop" if node % 2 == 0 else "explore"
        )
        fitness = -10.0 + 0.2 * node
        outcome = "improve" if node % 3 == 0 else "plateau"
        algorithms.append({"id": node, "parent_id": parent})
        rows.append(
            [
                slot, parent, node, action, "ordinary" if action else "initialization",
                outcome, "ok", fitness, None, "",
                0.3, 0.5, 1.5, 2.0, n_roots, 2, 1, "initial", slot,
                0.5, "",
            ]
        )

    with (run_dir / "evaluations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(OUTCOME_HEADER)
        for row in rows:
            writer.writerow(["" if value is None else value for value in row])

    events: list[dict[str, object]] = []
    for decision in range(n_ordinary):
        parent = decision + 1
        snapshot = [
            {
                "id": node,
                "q": 0.1 + 0.1 * (node % 5),
                "P": 0.5,
                "U": 0.5,
                "B": float(node % 3),
                "c_t": node % 4,
                "T": 0.5,
                "S": 0.4 + 0.05 * (node % 3),
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
                "beta": 1.0 + decision,
                "ess": 2.0,
                "marker": 0.42,
                "snapshot": snapshot,
            }
        )
        events.append(
            {
                "event": "action_decision",
                "decision_index": decision,
                "slot": n_roots + decision,
                "parent_id": parent,
                "T": 0.4 + 0.02 * decision,
                "p_explore": 0.3 + 0.01 * decision,
                "action": "develop" if decision % 2 == 0 else "explore",
                "action_draw": 0.2,
            }
        )
    with (run_dir / "mechanism_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")

    distances = np.abs(np.subtract.outer(np.arange(n_nodes), np.arange(n_nodes))).astype(
        np.float32
    )
    np.savez(
        run_dir / "checkpoints" / "behave.npz",
        ids=np.arange(1, n_nodes + 1, dtype=np.int64),
        matrix=distances,
        states_a=np.zeros(0, dtype=np.int32),
        lengths_a=np.zeros(0, dtype=np.int32),
        states_b=np.zeros(0, dtype=np.int32),
        lengths_b=np.zeros(0, dtype=np.int32),
    )
    (run_dir / "checkpoints" / "latest.json").write_text(
        json.dumps(
            {
                "weights": [0.75, 0.10, 0.15],
                "tree": {"algorithms": algorithms},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "logs").mkdir()
    (run_dir / "logs" / "summary.json").write_text(
        json.dumps(
            {
                "task": "tsp_construct",
                "budget_slots": n_roots + n_ordinary,
                "n_algorithms": n_nodes,
                "profiling_wall_time": 3.2,
                "repair_llm_calls": 0,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_rolling_fractions_partition_to_one() -> None:
    fractions = rolling_fractions(
        ["a", "a", "b", "a", "b", "b", "b"], ["a", "b"], window=3
    )
    totals = fractions["a"] + fractions["b"]
    assert np.allclose(totals, 1.0)
    assert fractions["a"][0] == 1.0
    assert fractions["b"][3] == pytest.approx(1 / 3)


def test_classical_mds_recovers_line_geometry() -> None:
    positions = np.array([[0.0, 0.0], [3.0, 0.0], [6.0, 0.0], [9.0, 0.0]])
    matrix = np.linalg.norm(positions[:, None] - positions[None], axis=2)
    coordinates = classical_mds(matrix)
    recovered = np.linalg.norm(coordinates[:, None] - coordinates[None], axis=2)
    assert np.allclose(recovered, matrix, atol=1e-6)


def test_neighborhood_edges_follow_frozen_k_formula() -> None:
    size = 20
    matrix = np.random.default_rng(0).random((size, size))
    matrix = (matrix + matrix.T) / 2
    edges = neighborhood_edges(matrix)
    assert len(edges) == size * 2
    assert all(left != right for left, right in edges)


def test_run_data_reads_live_artifacts(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    run = RunData(run_dir)
    assert run.task == "tsp_construct"
    assert run.n_nodes == 14
    assert len(run.rows) == 14
    assert len(run.action_decisions()) == 6
    assert sum(run.opportunities().values()) == 6
    assert run.matrix.shape == (14, 14)


def test_dashboard_and_landscape_figures_written(tmp_path: Path) -> None:
    run = RunData(_write_run(tmp_path))
    dashboard = plot_dashboard(
        run, tmp_path / "out" / "v9_19_run_dashboard.png"
    )
    landscape = plot_landscape(run, tmp_path / "out" / "v9_19_run_landscape.png")
    assert dashboard.is_file() and dashboard.stat().st_size > 20_000
    assert landscape is not None and landscape.is_file()


def test_task_curves_figure_written(tmp_path: Path) -> None:
    runs = [RunData(_write_run(tmp_path / f"r{index}")) for index in range(3)]
    output = plot_task_curves(runs, tmp_path / "out" / "curves.png")
    assert output.is_file() and output.stat().st_size > 20_000


def test_landscape_skipped_without_distance_matrix(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    (run_dir / "checkpoints" / "behave.npz").unlink()
    run = RunData(run_dir)
    assert plot_landscape(run, tmp_path / "out" / "x.png") is None


def test_dashboard_tolerates_partial_jsonl_line(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path)
    with (run_dir / "mechanism_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"event": "pre_deci')
    run = RunData(run_dir)
    assert len(run.pre_decisions()) == 6
    output = plot_dashboard(run, tmp_path / "out" / "partial.png")
    assert output.is_file()

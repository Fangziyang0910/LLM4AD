"""TraceAAD checkpoint dump / load.

Periodic full-state snapshots under ``logs/checkpoints/`` so a run can resume
without replaying incomplete event logs.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

from .curriculum import EliteCurriculum
from .derivation_graph import DerivationGraph
from .experience_memory import ExperienceMemory
from .feedback import RankingModel
from .portfolio import OperatorPortfolio, OperatorStats
from .schema import (
    ChampionEvent,
    ImprovementEdge,
    ProgramNode,
    Trajectory,
    TrajectoryStatus,
    ValueVec,
)
from .trajectory_memory import TrajectoryMemory

CHECKPOINT_VERSION = 1
CHECKPOINT_DIRNAME = "checkpoints"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _serialize_rng(rng) -> dict[str, Any]:
    version, state, gauss = rng.getstate()
    return {"version": version, "state": list(state), "gauss": gauss}


def _restore_rng(rng, payload: Mapping[str, Any]) -> None:
    rng.setstate((payload["version"], tuple(payload["state"]), payload.get("gauss")))


def _serialize_value_vec(value: ValueVec | None) -> dict[str, float] | None:
    if value is None:
        return None
    return asdict(value)


def _restore_value_vec(payload: Mapping[str, Any] | None) -> ValueVec | None:
    if payload is None:
        return None
    return ValueVec(**{f.name: float(payload.get(f.name, 0.0)) for f in fields(ValueVec)})


def serialize_graph(graph: DerivationGraph) -> dict[str, Any]:
    return {
        "next_node_id": graph._next_node_id,
        "next_edge_id": graph._next_edge_id,
        "nodes": [
            {
                "id": node.id,
                "code": node.code,
                "idea": node.idea,
                "fitness": node.fitness,
                "complexity": node.complexity,
                "runtime": node.runtime,
                "complexity_metrics": dict(node.complexity_metrics),
            }
            for node in graph.nodes()
        ],
        "edges": [
            {
                "id": edge.id,
                "parent_id": edge.parent_id,
                "child_id": edge.child_id,
                "action": edge.action,
                "operator": edge.operator,
                "delta": edge.delta,
                "outcome": edge.outcome,
                "iteration": edge.iteration,
            }
            for edge in graph.edges()
        ],
    }


def restore_graph(payload: Mapping[str, Any]) -> DerivationGraph:
    graph = DerivationGraph()
    for node_data in payload["nodes"]:
        node = ProgramNode(
            id=int(node_data["id"]),
            code=str(node_data["code"]),
            idea=str(node_data.get("idea") or ""),
            fitness=node_data.get("fitness"),
            complexity=float(node_data.get("complexity") or 0.0),
            runtime=float(node_data.get("runtime") or 0.0),
            complexity_metrics=dict(node_data.get("complexity_metrics") or {}),
        )
        graph._nodes[node.id] = node
    for edge_data in payload["edges"]:
        edge = ImprovementEdge(
            id=int(edge_data["id"]),
            parent_id=int(edge_data["parent_id"]),
            child_id=int(edge_data["child_id"]),
            action=str(edge_data["action"]),
            operator=str(edge_data.get("operator") or "unknown"),
            delta=edge_data.get("delta"),
            outcome=str(edge_data.get("outcome") or "unknown"),
            iteration=edge_data.get("iteration"),
        )
        graph._edges[edge.id] = edge
        graph._incoming_edge_by_child[edge.child_id] = edge.id
        graph._outgoing_edges_by_parent[edge.parent_id].append(edge.id)
    graph._next_node_id = int(payload["next_node_id"])
    graph._next_edge_id = int(payload["next_edge_id"])
    return graph


def serialize_memory(memory: TrajectoryMemory) -> dict[str, Any]:
    trajectories = []
    for traj in memory.trajectories():
        trajectories.append(
            {
                "id": traj.id,
                "node_ids": list(traj.node_ids),
                "edge_ids": list(traj.edge_ids),
                "endpoint_id": traj.endpoint_id,
                "base_id": traj.base_id,
                "island_id": traj.island_id,
                "visit_count": traj.visit_count,
                "status": str(traj.status),
                "value": _serialize_value_vec(traj.value),
                "scalar_value": traj.scalar_value,
            }
        )
    return {
        "max_trajectory_length": memory.max_trajectory_length,
        "next_id": memory._next_id,
        "trajectories": trajectories,
    }


def restore_memory(payload: Mapping[str, Any]) -> TrajectoryMemory:
    memory = TrajectoryMemory(max_trajectory_length=int(payload["max_trajectory_length"]))
    memory._next_id = int(payload["next_id"])
    for item in payload["trajectories"]:
        traj = Trajectory(
            id=int(item["id"]),
            node_ids=tuple(int(x) for x in item["node_ids"]),
            edge_ids=tuple(int(x) for x in item["edge_ids"]),
            endpoint_id=int(item["endpoint_id"]),
            base_id=int(item["base_id"]),
            island_id=int(item.get("island_id", 0)),
            visit_count=int(item.get("visit_count", 0)),
            status=TrajectoryStatus(item.get("status", TrajectoryStatus.ACTIVE)),
            value=_restore_value_vec(item.get("value")),
            scalar_value=item.get("scalar_value"),
        )
        memory._trajectories[traj.id] = traj
    return memory


def serialize_portfolio(portfolio: OperatorPortfolio) -> dict[str, Any]:
    stats = {}
    for name, s in portfolio.stats.items():
        stats[name] = {
            "attempt_count": s.attempt_count,
            "eligible_count": s.eligible_count,
            "discounted_mass": s.discounted_mass,
            "ema_utility": s.ema_utility,
            "ema_downside": s.ema_downside,
            "ema_valid": s.ema_valid,
            "ema_novel": s.ema_novel,
            "ema_cost": s.ema_cost,
            "ema_global_best": s.ema_global_best,
            "ema_near_record": s.ema_near_record,
        }
    return {
        "stats": stats,
        "updated_iterations": [
            [op_name, int(attempt_id)]
            for op_name, attempt_id in sorted(portfolio._updated_iterations)
        ],
    }


def restore_portfolio(portfolio: OperatorPortfolio, payload: Mapping[str, Any]) -> None:
    for name, stats_data in payload.get("stats", {}).items():
        if name not in portfolio.stats:
            portfolio.stats[name] = OperatorStats()
        s = portfolio.stats[name]
        s.attempt_count = int(stats_data.get("attempt_count", 0))
        s.eligible_count = int(stats_data.get("eligible_count", 0))
        s.discounted_mass = float(stats_data.get("discounted_mass", 0.0))
        s.ema_utility = stats_data.get("ema_utility")
        s.ema_downside = stats_data.get("ema_downside")
        s.ema_valid = stats_data.get("ema_valid")
        s.ema_novel = stats_data.get("ema_novel")
        s.ema_cost = stats_data.get("ema_cost")
        s.ema_global_best = stats_data.get("ema_global_best")
        s.ema_near_record = stats_data.get("ema_near_record")
    portfolio._updated_iterations = {
        (str(op_name), int(attempt_id))
        for op_name, attempt_id in payload.get("updated_iterations", [])
    }


def serialize_curriculum(curriculum: EliteCurriculum) -> dict[str, Any]:
    return {
        "maximize": curriculum._maximize,
        "max_champion_events": curriculum._max_champion_events,
        "max_positive_traces": curriculum._max_positive_traces,
        "packet_count": curriculum._packet_count,
        "usage": dict(curriculum._usage),
        "reward": dict(curriculum._reward),
        "champion_events": [asdict(event) for event in curriculum._champion_events],
    }


def restore_curriculum(
    graph: DerivationGraph,
    payload: Mapping[str, Any],
) -> EliteCurriculum:
    curriculum = EliteCurriculum(
        graph,
        maximize=bool(payload.get("maximize", True)),
        max_champion_events=int(payload.get("max_champion_events", 4)),
        max_positive_traces=int(payload.get("max_positive_traces", 2)),
    )
    curriculum._packet_count = int(payload.get("packet_count", 0))
    curriculum._usage = defaultdict(int, {k: int(v) for k, v in payload.get("usage", {}).items()})
    curriculum._reward = defaultdict(
        float, {k: float(v) for k, v in payload.get("reward", {}).items()}
    )
    curriculum._champion_events = [
        ChampionEvent(**event_data) for event_data in payload.get("champion_events", [])
    ]
    return curriculum


def serialize_ranking(ranking: RankingModel) -> dict[str, Any]:
    return {
        "k": ranking.k,
        "scores": {str(node_id): float(score) for node_id, score in ranking._scores.items()},
        "parents": {str(node_id): int(parent) for node_id, parent in ranking._parents.items()},
    }


def restore_ranking(payload: Mapping[str, Any]) -> RankingModel:
    ranking = RankingModel(k=float(payload.get("k", 16.0)))
    ranking._scores = defaultdict(
        float, {int(node_id): float(score) for node_id, score in payload.get("scores", {}).items()}
    )
    ranking._parents = {
        int(node_id): int(parent) for node_id, parent in payload.get("parents", {}).items()
    }
    return ranking


def dump_traceaad_state(method) -> dict[str, Any]:
    best = method._best_node
    return {
        "version": CHECKPOINT_VERSION,
        "tot_sample_nums": int(method._tot_sample_nums),
        "stagnation": int(getattr(method, "_stagnation", 0)),
        "best_node_id": None if best is None else int(best.id),
        "rng": _serialize_rng(method._rng),
        "graph": serialize_graph(method._graph),
        "memory": serialize_memory(method._memory),
        "portfolio": serialize_portfolio(method._portfolio),
        "curriculum": serialize_curriculum(method._curriculum),
        "ranking": serialize_ranking(method._ranking),
    }


def load_traceaad_state(method, payload: Mapping[str, Any]) -> None:
    version = int(payload.get("version", 0))
    if version != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported TraceAAD checkpoint version: {version}")

    graph = restore_graph(payload["graph"])
    memory = restore_memory(payload["memory"])
    if memory.max_trajectory_length != method._memory.max_trajectory_length:
        # Keep constructor config; loaded paths already truncated with old window.
        pass

    method._graph = graph
    method._memory = memory
    method._experience_memory = ExperienceMemory(graph)
    method._curriculum = restore_curriculum(graph, payload["curriculum"])
    method._ranking = restore_ranking(payload["ranking"])
    restore_portfolio(method._portfolio, payload["portfolio"])
    _restore_rng(method._rng, payload["rng"])

    method._tot_sample_nums = int(payload["tot_sample_nums"])
    method._stagnation = int(payload.get("stagnation", 0))
    best_id = payload.get("best_node_id")
    method._best_node = None if best_id is None else graph.get_node(int(best_id))
    method._batch_cost = 0.0
    method._batch_candidate_attempts = 0
    # search_iteration renumbers from 0 after resume; drop per-run dedupe keys.
    method._portfolio._updated_iterations.clear()
    method._resume_mode = True


def checkpoint_dir(log_dir: str | Path) -> Path:
    return Path(log_dir) / CHECKPOINT_DIRNAME


def checkpoint_path(log_dir: str | Path, sample_order: int) -> Path:
    return checkpoint_dir(log_dir) / f"ckpt_{int(sample_order)}.json"


def latest_checkpoint_path(log_dir: str | Path) -> Path:
    return checkpoint_dir(log_dir) / "latest.json"


def save_checkpoint(method, log_dir: str | Path | None = None) -> Path | None:
    """Write ``ckpt_{N}.json`` and refresh ``latest.json``. Returns path or None."""
    resolved = log_dir
    if resolved is None:
        profiler = getattr(method, "_profiler", None)
        resolved = None if profiler is None else getattr(profiler, "_log_dir", None)
    if not resolved:
        return None
    sample_order = int(method._tot_sample_nums)
    path = checkpoint_path(resolved, sample_order)
    payload = dump_traceaad_state(method)
    payload["sample_order"] = sample_order
    _atomic_write_json(path, payload)
    _atomic_write_json(latest_checkpoint_path(resolved), payload)
    return path


def find_latest_checkpoint(log_dir: str | Path) -> Path:
    log_path = Path(log_dir)
    latest = latest_checkpoint_path(log_path)
    if latest.is_file():
        return latest
    ckpt_root = checkpoint_dir(log_path)
    if not ckpt_root.is_dir():
        raise FileNotFoundError(f"no TraceAAD checkpoints under {ckpt_root}")
    orders: list[tuple[int, Path]] = []
    for path in ckpt_root.glob("ckpt_*.json"):
        match = re.fullmatch(r"ckpt_(\d+)\.json", path.name)
        if match:
            orders.append((int(match.group(1)), path))
    if not orders:
        raise FileNotFoundError(f"no TraceAAD checkpoints under {ckpt_root}")
    return max(orders, key=lambda item: item[0])[1]


def load_checkpoint(method, path: str | Path) -> Path:
    ckpt = Path(path)
    with ckpt.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    load_traceaad_state(method, payload)
    return ckpt

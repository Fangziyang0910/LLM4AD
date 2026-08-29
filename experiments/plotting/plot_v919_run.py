"""V9.19 training-process visualization.

Two figures per run, driven by the online quantities V9.19 records: the
per-slot landscape/trajectory/action events in ``mechanism_events.jsonl``,
the attempt ledger in ``evaluations.csv``, and the combined BehaveSim
distance matrix in ``checkpoints/behave.npz``.

- ``v9_19_<run>_dashboard.png``  search progress, outcome and action
  mixtures, the P/U/T competition behind each parent choice, trajectory
  response versus explore probability, and allocation temperature.
- ``v9_19_<run>_landscape.png``  the behavior landscape at the last
  checkpoint: classical-MDS layout of the distance matrix with formation
  edges versus behavior neighborhoods, the frontier's growth over time, and
  how granted opportunities spread over quality levels.

``--task`` mode plots the three repeats' search curves of one task.  All
modes read live artifacts and tolerate in-progress runs.

Examples::

    uv run python experiments/plotting/plot_v919_run.py \
        --run-dir experiments/tsp_construct/traceaad_v9_19/v9_19_tsp_construct_rep1
    uv run python experiments/plotting/plot_v919_run.py --task tsp_construct
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm4ad.method.traceaad_v9_19.schema import (  # noqa: E402
    MIN_NEIGHBORS,
    NEIGHBORHOOD_FRACTION,
)

OUTCOME_COLORS = {
    "improve": "#009E73",
    "plateau": "#8C8C8C",
    "regress": "#E69F00",
    "invalid": "#CC79A7",
    "duplicate": "#B0BEC5",
    "timeout": "#D55E00",
    "profiling_failed": "#000000",
}
ACTION_COLORS = {
    "develop": "#0072B2",
    "explore": "#D55E00",
}
LAYER_COLORS = {"P": "#264653", "U": "#E9C46A", "T": "#F4A261"}

plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.15,
    }
)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


class RunData:
    """Everything the figures need from one run directory (live-safe)."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.name = run_dir.name
        self.events = _load_events(run_dir / "mechanism_events.jsonl")
        self.rows = _load_rows(run_dir / "evaluations.csv")
        self.summary = _load_json(run_dir / "logs" / "summary.json")
        checkpoint = _load_json(run_dir / "checkpoints" / "latest.json")
        self.weights = checkpoint.get("weights", [0.75, 0.10, 0.15])
        self.parents: dict[int, int | None] = {}
        for item in checkpoint.get("tree", {}).get("algorithms", []):
            self.parents[int(item["id"])] = item.get("parent_id")
        self.matrix: np.ndarray | None = None
        self.matrix_ids: list[int] = []
        behave = run_dir / "checkpoints" / "behave.npz"
        if behave.is_file():
            with np.load(behave) as payload:
                self.matrix = np.asarray(payload["matrix"], dtype=np.float64)
                self.matrix_ids = [int(v) for v in payload["ids"].tolist()]

    @property
    def n_nodes(self) -> int:
        """Node count from the summary, else the live checkpoint state."""
        if isinstance(self.summary.get("n_algorithms"), int):
            return self.summary["n_algorithms"]
        return sum(1 for node_id in self.parents if node_id)

    @property
    def task(self) -> str:
        return str(self.summary.get("task", "")) or _task_from_path(self.run_dir)

    def pre_decisions(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == "pre_decision"]

    def action_decisions(self) -> list[dict[str, Any]]:
        return [e for e in self.events if e.get("event") == "action_decision"]

    def opportunities(self) -> Counter[int]:
        counts: Counter[int] = Counter()
        for row in self.rows:
            parent = int(row["parent_id"])
            if parent:
                counts[parent] += 1
        return counts


def _task_from_path(run_dir: Path) -> str:
    for part in reversed(run_dir.parts):
        if part in {
            "tsp_construct",
            "cvrp_aco",
            "op_aco",
            "online_bin_packing",
            "vrptw_construct",
        }:
            return part
    return "unknown"


def _load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # an in-progress run may end mid-line
    return events


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
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rolling_fractions(
    labels: list[str], categories: list[str], window: int
) -> dict[str, np.ndarray]:
    """Rolling share of each category over a label axis."""
    fractions = {
        category: np.full(len(labels), np.nan) for category in categories
    }
    codes = {category: index for index, category in enumerate(categories)}
    coded = np.array(
        [codes.get(label, -1) for label in labels], dtype=np.int64
    )
    for position in range(len(labels)):
        lo = max(0, position - window + 1)
        block = coded[lo : position + 1]
        block = block[block >= 0]
        if block.size == 0:
            continue
        counts = np.bincount(block, minlength=len(categories)).astype(float)
        shares = counts / counts.sum()
        for index, category in enumerate(categories):
            fractions[category][position] = shares[index]
    return fractions


def classical_mds(matrix: np.ndarray) -> np.ndarray:
    """Deterministic two-dimensional classical MDS of a distance matrix."""
    squared = matrix ** 2
    size = len(matrix)
    centerer = np.eye(size) - np.ones((size, size)) / size
    centered = -0.5 * centerer @ squared @ centerer
    centered = (centered + centered.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(centered)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order][:2], 0.0, None)
    coordinates = eigenvectors[:, order[:2]] * np.sqrt(eigenvalues)
    return coordinates


def neighborhood_edges(matrix: np.ndarray) -> list[tuple[int, int]]:
    size = len(matrix)
    if size < 2:
        return []
    k = min(size - 1, max(MIN_NEIGHBORS, math.ceil(NEIGHBORHOOD_FRACTION * (size - 1))))
    filled = matrix.copy()
    np.fill_diagonal(filled, np.inf)
    edges: list[tuple[int, int]] = []
    for index in range(size):
        order = np.argsort(filled[index], kind="stable")
        for neighbor in order[:k]:
            edges.append((index, int(neighbor)))
    return edges


def _slot_fitness(rows: list[dict[str, str]]) -> tuple[list[int], list[float]]:
    slots: list[int] = []
    values: list[float] = []
    for row in rows:
        raw = row.get("fitness", "")
        if not raw:
            continue
        slots.append(int(row["slot"]))
        values.append(float(raw))
    return slots, values


def _best_curve(slots: list[int], values: list[float]) -> tuple[list[int], list[float]]:
    best_slots: list[int] = []
    best_values: list[float] = []
    running = -math.inf
    for slot, value in sorted(zip(slots, values)):
        if value > running:
            running = value
            best_slots.append(slot)
            best_values.append(value)
    return best_slots, best_values


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def plot_dashboard(run: RunData, out_path: Path) -> Path:
    rows = run.rows
    if not rows:
        raise ValueError(f"no evaluations recorded under {run.run_dir}")
    outcomes = [row["outcome"] for row in rows]
    fitness_slots, fitness_values = _slot_fitness(rows)
    window = max(5, len(rows) // 20)

    figure, axes = plt.subplots(2, 3, figsize=(11.5, 6.2))

    # (a) search progress
    ax = axes[0][0]
    per_action: dict[str, tuple[list[int], list[float]]] = {}
    for row in rows:
        raw = row.get("fitness", "")
        if not raw:
            continue
        slot, value = int(row["slot"]), float(raw)
        bucket = per_action.setdefault(row["action"] or "init", ([], []))
        bucket[0].append(slot)
        bucket[1].append(value)
    for name, (xs, ys) in sorted(per_action.items()):
        color = ACTION_COLORS.get(name, "#8C8C8C")
        ax.scatter(xs, ys, s=9, color=color, label=name, alpha=0.75, zorder=2)
    if fitness_values:
        best_slots, best_values = _best_curve(fitness_slots, fitness_values)
        ax.plot(
            best_slots, best_values, color="#111111", lw=1.6, zorder=3, label="best"
        )
    ax.set_xlabel("primary slot")
    ax.set_ylabel("fitness")
    ax.set_title("(a) search progress")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)

    # (b) outcome mixture
    ax = axes[0][1]
    categories = [c for c in OUTCOME_COLORS if c in set(outcomes)]
    fractions = rolling_fractions(outcomes, categories, window)
    axis = np.arange(len(outcomes))
    stack = [np.nan_to_num(fractions[c], nan=0.0) for c in categories]
    ax.stackplot(
        axis,
        stack,
        labels=categories,
        colors=[OUTCOME_COLORS[c] for c in categories],
        alpha=0.9,
    )
    ax.set_xlim(0, max(0, len(outcomes) - 1))
    ax.set_ylim(0, 1)
    ax.set_xlabel("settled attempt")
    ax.set_ylabel("rolling share")
    ax.set_title(f"(b) outcomes (window={window})")
    ax.legend(loc="upper right", ncol=2)

    # (c) action mixture (ordinary slots only)
    ax = axes[0][2]
    ordinary = [row for row in rows if row["action"]]
    ordinary_actions = [row["action"] for row in ordinary]
    if ordinary_actions:
        categories = [c for c in ACTION_COLORS if c in set(ordinary_actions)]
        fractions = rolling_fractions(ordinary_actions, categories, window)
        axis = np.arange(len(ordinary_actions))
        stack = [np.nan_to_num(fractions[c], nan=0.0) for c in categories]
        ax.stackplot(
            axis,
            stack,
            labels=categories,
            colors=[ACTION_COLORS[c] for c in categories],
            alpha=0.9,
        )
        ax.set_xlim(0, max(0, len(ordinary_actions) - 1))
    ax.set_ylim(0, 1)
    ax.set_xlabel("ordinary attempt")
    ax.set_ylabel("rolling share")
    ax.set_title("(c) action mixture")
    ax.legend(loc="upper right", ncol=1)

    # (d) P/U/T competition behind each parent choice
    ax = axes[1][0]
    selected = _selected_snapshots(run)
    if selected:
        w_p, w_u, w_t = run.weights
        xs = np.arange(len(selected))
        parts = [
            ("P", [w_p * s["P"] for s in selected]),
            ("U", [w_u * s["U"] for s in selected]),
            ("T", [w_t * s["T"] for s in selected]),
        ]
        ax.stackplot(
            xs,
            [part[1] for part in parts],
            labels=[part[0] for part in parts],
            colors=[LAYER_COLORS[part[0]] for part in parts],
            alpha=0.9,
        )
        ax.set_xlim(0, len(selected) - 1)
        ax.legend(loc="lower right", ncol=3)
    ax.set_xlabel("ordinary decision")
    ax.set_ylabel("weighted contribution")
    ax.set_title("(d) selected parent: S decomposition")

    # (e) trajectory response vs explore probability
    ax = axes[1][1]
    decisions = run.action_decisions()
    if decisions:
        xs = [event["decision_index"] for event in decisions]
        ax.plot(
            xs,
            [event["T"] for event in decisions],
            color="#264653",
            lw=1.6,
            marker="o",
            ms=2.5,
            label="T(a)",
        )
        ax.plot(
            xs,
            [event["p_explore"] for event in decisions],
            color="#D55E00",
            lw=1.6,
            marker="s",
            ms=2.5,
            label="p_E",
        )
        ax.axhline(0.5, color="#8C8C8C", lw=0.7, ls=":")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", ncol=2)
    ax.set_xlabel("ordinary decision")
    ax.set_ylabel("value")
    ax.set_title("(e) trajectory response T vs explore prob p_E")

    # (f) allocation temperature
    ax = axes[1][2]
    pre = run.pre_decisions()
    if pre:
        xs = [event["decision_index"] for event in pre]
        betas = [event["beta"] for event in pre]
        ax.plot(xs, betas, color="#0072B2", label=r"$\beta$")
        if any(beta > 0 for beta in betas):
            ax.set_yscale("log")
        ax.set_ylabel(r"inverse temperature $\beta$")
        twin = ax.twinx()
        twin.spines["right"].set_visible(True)
        twin.grid(False)
        twin.plot(xs, [event["ess"] for event in pre], color="#009E73", label="ESS")
        twin.plot(
            xs, [event["pool_size"] for event in pre], color="#E69F00", label="M"
        )
        twin.set_ylabel("ESS / pool size M")
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = twin.get_legend_handles_labels()
        ax.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    ax.set_xlabel("ordinary decision")
    ax.set_title("(f) allocation temperature")

    profiling = run.summary.get("profiling_wall_time")
    repairs = run.summary.get("repair_llm_calls")
    footer = run.summary.get("budget_slots", len(rows))
    figure.suptitle(
        f"{run.task} | {run.name} | slots={footer} nodes={run.n_nodes}"
        + (f" | profiling={profiling:.0f}s" if isinstance(profiling, (int, float)) else "")
        + (f" | repairs={repairs}" if repairs else ""),
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def _selected_snapshots(run: RunData) -> list[dict[str, Any]]:
    """The selected parent's snapshot entry per ordinary decision."""
    snapshot_by_decision = {
        event["decision_index"]: {
            entry["id"]: entry for entry in event.get("snapshot", [])
        }
        for event in run.pre_decisions()
    }
    selected: list[dict[str, Any]] = []
    for event in run.action_decisions():
        index = event["decision_index"]
        entry = snapshot_by_decision.get(index, {}).get(event["parent_id"])
        if entry is not None:
            selected.append(entry)
    return selected


def plot_landscape(run: RunData, out_path: Path) -> Path | None:
    if run.matrix is None or len(run.matrix_ids) < 2:
        return None
    matrix = run.matrix
    coordinates = classical_mds(matrix)
    ids = run.matrix_ids
    index_of = {node_id: position for position, node_id in enumerate(ids)}

    fitness = {
        int(row["child_id"]): float(row["fitness"])
        for row in run.rows
        if row["child_id"] and row["fitness"]
    }
    quality = {node_id: fitness.get(node_id) for node_id in ids}
    known = [q for q in quality.values() if q is not None]
    ranks = {
        node_id: (
            None
            if quality[node_id] is None
            else _percentile(quality[node_id], known)
        )
        for node_id in ids
    }
    created = {
        int(row["child_id"]): int(row["slot"])
        for row in run.rows
        if row["child_id"]
    }
    opportunities = run.opportunities()
    creator = {
        int(row["child_id"]): row["action"] or "init"
        for row in run.rows
        if row["child_id"]
    }

    formation_edges = [
        (index_of[node_id], index_of[parent_id])
        for node_id, parent_id in run.parents.items()
        if node_id in index_of and parent_id in index_of
    ]
    neighbor_edges = neighborhood_edges(matrix)

    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))

    # (a) quality view with formation edges and behavior neighborhoods
    ax = axes[0]
    for i, j in neighbor_edges:
        ax.plot(
            [coordinates[i, 0], coordinates[j, 0]],
            [coordinates[i, 1], coordinates[j, 1]],
            color="#B0BEC5",
            lw=0.5,
            alpha=0.35,
            zorder=1,
        )
    for i, j in formation_edges:
        ax.plot(
            [coordinates[i, 0], coordinates[j, 0]],
            [coordinates[i, 1], coordinates[j, 1]],
            color="#264653",
            lw=0.7,
            alpha=0.55,
            zorder=2,
        )
    _scatter_landscape(
        ax,
        coordinates,
        [ranks[node_id] for node_id in ids],
        opportunities,
        ids,
        cmap="viridis",
    )
    ax.set_title("(a) landscape: Q (gray=behavior kNN, dark=formation)")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")

    # (b) temporal view: how the frontier expanded
    ax = axes[1]
    slots_sorted = sorted(created.values())
    if slots_sorted:
        slot_ranks = {
            node_id: _percentile(created[node_id], slots_sorted) for node_id in ids
        }
        for i, j in formation_edges:
            ax.plot(
                [coordinates[i, 0], coordinates[j, 0]],
                [coordinates[i, 1], coordinates[j, 1]],
                color="#8C8C8C",
                lw=0.6,
                alpha=0.5,
                zorder=1,
            )
        _scatter_landscape(
            ax, coordinates, [slot_ranks[node_id] for node_id in ids],
            opportunities, ids, cmap="magma",
        )
    ax.set_title("(b) landscape: creation slot")
    ax.set_xlabel("MDS 1")

    # (c) allocation: granted opportunities versus quality percentile
    ax = axes[2]
    for node_id in ids:
        if ranks[node_id] is None:
            continue
        action = creator.get(node_id, "init")
        ax.scatter(
            ranks[node_id],
            opportunities.get(node_id, 0),
            s=18,
            color=ACTION_COLORS.get(action, "#8C8C8C"),
            alpha=0.8,
        )
    ax.set_xlabel("quality percentile Q (last snapshot)")
    ax.set_ylabel("opportunities c_t")
    ax.set_title("(c) allocation vs quality")

    figure.suptitle(
        f"{run.task} | {run.name} | nodes={len(ids)}", fontsize=10
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


def _scatter_landscape(
    ax,
    coordinates: np.ndarray,
    color_values: list[float | None],
    opportunities: Counter[int],
    ids: list[int],
    *,
    cmap: str,
) -> None:
    sizes = [14 + 2.4 * math.sqrt(opportunities.get(node_id, 0)) for node_id in ids]
    known_x = [i for i, v in enumerate(color_values) if v is not None]
    if known_x:
        scatter = ax.scatter(
            coordinates[known_x, 0],
            coordinates[known_x, 1],
            c=[color_values[i] for i in known_x],
            s=[sizes[i] for i in known_x],
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            zorder=3,
            edgecolors="white",
            linewidths=0.3,
        )
        figure = ax.get_figure()
        figure.colorbar(scatter, ax=ax, shrink=0.75, pad=0.02)
    unknown_x = [i for i, v in enumerate(color_values) if v is None]
    if unknown_x:
        ax.scatter(
            coordinates[unknown_x, 0],
            coordinates[unknown_x, 1],
            s=[sizes[i] for i in unknown_x],
            color="#B0BEC5",
            zorder=3,
        )


def _percentile(value: float, pool: list[float]) -> float:
    pool = sorted(pool)
    below = sum(1 for item in pool if item < value)
    span = max(1, len(pool) - 1)
    return below / span


# ---------------------------------------------------------------------------
# task-level curves
# ---------------------------------------------------------------------------


def plot_task_curves(runs: list[RunData], out_path: Path) -> Path:
    figure, axes = plt.subplots(
        1, 2, figsize=(7.6, 3.0)
    )
    budget = max(
        (int(row["slot"]) for run in runs for row in run.rows), default=0
    )

    # (a) best-fitness curves per repeat
    ax = axes[0]
    colors = ["#0072B2", "#E69F00", "#009E73"]
    curves = []
    for index, run in enumerate(sorted(runs, key=lambda r: r.name)):
        slots, values = _slot_fitness(run.rows)
        if not values:
            continue
        best_slots, best_values = _best_curve(slots, values)
        color = colors[index % len(colors)]
        ax.plot(best_slots, best_values, color=color, lw=1.4, label=run.name)
        curves.append((best_slots, best_values))
    if curves:
        grid = np.arange(1, budget + 1)
        interpolated = []
        for best_slots, best_values in curves:
            stepped = np.full(len(grid), np.nan)
            position = np.searchsorted(np.asarray(best_slots), grid, side="right") - 1
            valid = position >= 0
            stepped[valid] = np.asarray(best_values)[position[valid]]
            interpolated.append(stepped)
        stacked = np.vstack(interpolated)
        mean = np.nanmean(stacked, axis=0)
        std = np.nanstd(stacked, axis=0)
        ok = np.isfinite(mean)
        ax.plot(
            grid[ok], mean[ok], color="#111111", lw=1.8, label="mean", zorder=4
        )
        ax.fill_between(
            grid[ok],
            (mean - std)[ok],
            (mean + std)[ok],
            color="#111111",
            alpha=0.12,
            zorder=1,
        )
    ax.set_xlabel("primary slot")
    ax.set_ylabel("best fitness")
    ax.set_title("(a) best fitness per repeat")
    ax.legend(loc="lower right")

    # (b) pooled outcome mixture across repeats
    ax = axes[1]
    pooled_rows = [row for run in runs for row in run.rows]
    outcomes = [row["outcome"] for row in pooled_rows]
    window = max(5, len(pooled_rows) // 20)
    categories = [c for c in OUTCOME_COLORS if c in set(outcomes)]
    fractions = rolling_fractions(outcomes, categories, window)
    axis = np.arange(len(outcomes))
    stack = [np.nan_to_num(fractions[c], nan=0.0) for c in categories]
    ax.stackplot(
        axis,
        stack,
        labels=categories,
        colors=[OUTCOME_COLORS[c] for c in categories],
        alpha=0.9,
    )
    ax.set_xlim(0, max(0, len(outcomes) - 1))
    ax.set_ylim(0, 1)
    ax.set_xlabel("settled attempt (pooled)")
    ax.set_ylabel("rolling share")
    ax.set_title(f"(b) outcome mixture (window={window})")
    ax.legend(loc="upper right", ncol=2)

    figure.suptitle(f"{runs[0].task} | {len(runs)} repeats", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path)
    plt.close(figure)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_dirs_for_task(task: str, experiments_root: Path) -> list[Path]:
    method_root = experiments_root / task / "traceaad_v9_19"
    return sorted(path for path in method_root.glob("v9_19_*") if path.is_dir())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append")
    parser.add_argument("--task")
    parser.add_argument("--experiments-root", type=Path, default=REPO_ROOT / "experiments")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)

    written: list[Path] = []
    if args.run_dir:
        for run_dir in args.run_dir:
            run = RunData(run_dir)
            output_dir = args.output_dir or run_dir / "figures"
            written.append(
                plot_dashboard(run, output_dir / f"v9_19_{run.name}_dashboard.png")
            )
            landscape = plot_landscape(
                run, output_dir / f"v9_19_{run.name}_landscape.png"
            )
            if landscape is not None:
                written.append(landscape)
    if args.task:
        dirs = run_dirs_for_task(args.task, args.experiments_root)
        if not dirs:
            raise SystemExit(f"no v9_19 run directories for task {args.task}")
        runs = [RunData(path) for path in dirs]
        output_dir = args.output_dir or dirs[0].parent / "figures"
        written.append(plot_task_curves(runs, output_dir / f"v9_19_{args.task}_curves.png"))
    if not args.run_dir and not args.task:
        parser.error("provide --run-dir or --task")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.instance_parallel import evaluate_instances, validate_backend
from llm4ad.task.optimization.main.tsp_aco.dataset import (
    DEFAULT_SPLIT,
    load_split_instances,
)
from llm4ad.task.optimization.main.tsp_aco.template import template_program, task_description

__all__ = ["TSPACOEvaluation"]


class ACO:
    """Ant colony optimizer adapted from the ReEvo TSP-ACO benchmark."""

    def __init__(
            self,
            distances,
            heuristic,
            n_ants=30,
            decay=0.9,
            alpha=1,
            beta=1,
            rng: np.random.Generator | None = None,
    ):
        self.problem_size = len(distances)
        self.distances = np.asarray(distances, dtype=float)
        self.heuristic = np.asarray(heuristic, dtype=float)
        self.n_ants = int(n_ants)
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.pheromone = np.ones_like(self.distances)
        self.shortest_path = None
        self.lowest_cost = float("inf")
        self.rng = rng if rng is not None else np.random.default_rng()

    def run(self, n_iterations: int) -> float:
        for _ in range(int(n_iterations)):
            paths = self.gen_path(require_prob=False)
            costs = self.gen_path_costs(paths)

            best_idx = int(np.argmin(costs))
            best_cost = float(costs[best_idx])
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost

            self.update_pheronome(paths, costs)

        return float(self.lowest_cost)

    def update_pheronome(self, paths, costs) -> None:
        self.pheromone = self.pheromone * self.decay
        for ant_idx in range(self.n_ants):
            path = paths[:, ant_idx]
            cost = costs[ant_idx]
            prev = np.roll(path, shift=1)
            self.pheromone[path, prev] += 1.0 / cost
            self.pheromone[prev, path] += 1.0 / cost

    def gen_path_costs(self, paths):
        assert paths.shape == (self.problem_size, self.n_ants)
        u = paths.T
        v = np.roll(u, shift=1, axis=1)
        return np.sum(self.distances[u, v], axis=1)

    def gen_path(self, require_prob=False):
        start = self.rng.integers(low=0, high=self.problem_size, size=self.n_ants)
        mask = np.ones((self.n_ants, self.problem_size), dtype=float)
        mask[np.arange(self.n_ants), start] = 0

        paths_list = [start]
        log_probs_list = []

        prev = start
        for _ in range(self.problem_size - 1):
            actions, log_probs = self.pick_move(prev, mask, require_prob)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                mask = mask.copy()
            prev = actions
            mask[np.arange(self.n_ants), actions] = 0

        if require_prob:
            return np.stack(paths_list), np.stack(log_probs_list)
        return np.stack(paths_list)

    def pick_move(self, prev, mask, require_prob):
        pheromone = self.pheromone[prev]
        heuristic = self.heuristic[prev]
        weights = ((pheromone ** self.alpha) * (heuristic ** self.beta) * mask)
        row_sums = weights.sum(axis=1)
        if np.any(row_sums <= 0) or not np.all(np.isfinite(row_sums)):
            raise ValueError("ACO transition probabilities are invalid.")

        actions = np.empty(self.n_ants, dtype=int)
        log_probs = np.empty(self.n_ants, dtype=float) if require_prob else None
        for ant_idx in range(self.n_ants):
            probs = weights[ant_idx] / row_sums[ant_idx]
            actions[ant_idx] = self.rng.choice(self.problem_size, p=probs)
            if require_prob:
                log_probs[ant_idx] = np.log(probs[actions[ant_idx]])
        return actions, log_probs


def distance_matrix(node_positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(node_positions, dtype=float)
    return np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)


def solve(
        node_positions: np.ndarray,
        heuristic: Callable,
        n_ants: int,
        n_iterations: int,
        rng: np.random.Generator,
) -> float:
    dist_mat = distance_matrix(node_positions)
    dist_mat[np.diag_indices_from(dist_mat)] = 1.0

    heu = np.asarray(heuristic(dist_mat.copy()), dtype=float)
    if heu.shape != dist_mat.shape or not np.all(np.isfinite(heu)):
        return float("inf")
    heu = heu + 1e-9
    heu[heu < 1e-9] = 1e-9

    aco = ACO(dist_mat, heu, n_ants=n_ants, rng=rng)
    return float(aco.run(n_iterations))


def _evaluate_tsp_aco_instance(heuristic: Callable, payload, context) -> float:
    idx, node_positions = payload
    rng = np.random.default_rng(None if context["seed"] is None else int(context["seed"]) + idx)
    try:
        return solve(
            node_positions=node_positions,
            heuristic=heuristic,
            n_ants=context["n_ants"],
            n_iterations=context["n_iterations"],
            rng=rng,
        )
    except Exception:
        return float("inf")


class TSPACOEvaluation(Evaluation):
    """Evaluator for TSP ant colony optimization heuristic matrices."""

    def __init__(
            self,
            timeout_seconds=120,
            split: str = DEFAULT_SPLIT,
            n_ants: int = 30,
            n_iterations: int = 100,
            seed: int | None = 1234,
            eval_workers: int = 1,
            eval_backend: str = "sequential",
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._datasets, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.problem_size = int(self.dataset_metadata["problem_size"])
        self.n_ants = int(n_ants)
        self.n_iterations = int(n_iterations)
        self.seed = seed
        self.eval_workers = max(1, int(eval_workers))
        self.eval_backend = validate_backend(eval_backend, daemon_eval_process=self.daemon_eval_process)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        try:
            costs = evaluate_instances(
                program_str=program_str,
                callable_func=callable_func,
                payloads=list(enumerate(self._datasets)),
                instance_eval=_evaluate_tsp_aco_instance,
                context={
                    "n_ants": self.n_ants,
                    "n_iterations": self.n_iterations,
                    "seed": self.seed,
                },
                backend=self.eval_backend,
                workers=self.eval_workers,
                timeout_seconds=self.timeout_seconds,
            )
            return -float(np.mean(costs))
        except Exception:
            return None

    def evaluate(self, heuristic: Callable) -> float:
        costs = []
        for idx, node_positions in enumerate(self._datasets):
            rng = np.random.default_rng(None if self.seed is None else int(self.seed) + idx)
            try:
                costs.append(solve(
                    node_positions=node_positions,
                    heuristic=heuristic,
                    n_ants=self.n_ants,
                    n_iterations=self.n_iterations,
                    rng=rng,
                ))
            except Exception:
                costs.append(float("inf"))

        return -float(np.mean(costs))

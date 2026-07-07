from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.instance_parallel import evaluate_instances, validate_backend
from llm4ad.task.optimization.main.op_aco.dataset import (
    DEFAULT_SPLIT,
    load_split_instances,
)
from llm4ad.task.optimization.main.op_aco.template import template_program, task_description

__all__ = ["OPACOEvaluation"]


def distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    distances = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
    distances[np.diag_indices_from(distances)] = 1e9
    return distances


def prizes(coordinates: np.ndarray) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    depot = coords[0]
    dists = np.linalg.norm(coords - depot, axis=1)
    result = 1 + np.floor(99 * dists / dists.max())
    result /= result.max()
    return result


def max_length(problem_size: int) -> float:
    for threshold, result in zip([50, 100, 200, 300], [3.0, 4.0, 5.0, 6.0]):
        if problem_size <= threshold:
            return result
    return 7.0


class ACO:
    """OP ant colony optimizer adapted from the HSEvo/ReEvo benchmark."""

    def __init__(
            self,
            node_prizes: np.ndarray,
            distances: np.ndarray,
            max_len: float,
            heuristic: np.ndarray,
            n_ants=20,
            decay=0.9,
            alpha=1,
            beta=1,
            rng: np.random.Generator | None = None,
    ):
        self.n = len(node_prizes)
        self.distances = np.asarray(distances, dtype=float)
        self.prizes = np.asarray(node_prizes, dtype=float)
        self.max_len = float(max_len)
        self.heuristic = np.asarray(heuristic, dtype=float)
        self.n_ants = int(n_ants)
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.pheromone = np.ones_like(self.distances)
        self.Q = 1.0 / self.prizes.sum()
        self.alltime_best_sol = None
        self.alltime_best_obj = 0.0
        self.rng = rng if rng is not None else np.random.default_rng()
        self.add_dummy_node()

    def add_dummy_node(self) -> None:
        self.prizes = np.concatenate((self.prizes, np.array([1e-10])))
        distances = np.concatenate((self.distances, 1e10 * np.ones((1, self.n))), axis=0)
        self.distances = np.concatenate((distances, 1e-10 + np.zeros((self.n + 1, 1))), axis=1)

        self.heuristic = np.concatenate((self.heuristic, np.zeros((1, self.n))), axis=0)
        self.heuristic = np.concatenate((self.heuristic, np.ones((self.n + 1, 1))), axis=1)
        self.pheromone = np.ones_like(self.distances)
        self.distances[self.distances == 1e-10] = 0
        self.prizes[-1] = 0

    def run(self, n_iterations: int) -> tuple[float, np.ndarray | None]:
        for _ in range(int(n_iterations)):
            sols = self.gen_sol()
            objs = self.gen_sol_obj(sols)
            sols = sols.T
            best_idx = int(np.argmax(objs))
            best_obj = float(objs[best_idx])
            if best_obj > self.alltime_best_obj:
                self.alltime_best_obj = best_obj
                self.alltime_best_sol = sols[best_idx]
            self.update_pheronome(sols, objs)
        return float(self.alltime_best_obj), self.alltime_best_sol

    def update_pheronome(self, sols, objs) -> None:
        self.pheromone = self.pheromone * self.decay
        for ant_idx in range(self.n_ants):
            sol = sols[ant_idx]
            obj = objs[ant_idx]
            nxt = np.roll(sol, shift=-1)
            self.pheromone[sol[:-1], nxt[:-1]] += self.Q * obj

    def gen_sol_obj(self, solutions):
        return self.prizes[solutions.T].sum(axis=1)

    def gen_sol(self):
        solutions = [np.zeros(self.n_ants, dtype=int)]
        mask = np.ones((self.n_ants, self.n + 1), dtype=float)
        travel_dis = np.zeros(self.n_ants, dtype=float)
        cur_node = np.zeros(self.n_ants, dtype=int)

        mask = self.update_mask(travel_dis, cur_node, mask)
        done = self.check_done(mask)
        while not done:
            nxt_node = self.pick_node(mask, cur_node)
            solutions.append(nxt_node)
            travel_dis += self.distances[cur_node, nxt_node]
            cur_node = nxt_node
            mask = self.update_mask(travel_dis, cur_node, mask)
            done = self.check_done(mask)
        return np.stack(solutions)

    def pick_node(self, mask, cur_node):
        pheromone = self.pheromone[cur_node]
        heuristic = self.heuristic[cur_node]
        weights = ((pheromone ** self.alpha) * (heuristic ** self.beta) * mask)
        row_sums = weights.sum(axis=1)
        if np.any(row_sums <= 0) or not np.all(np.isfinite(row_sums)):
            raise ValueError("ACO transition probabilities are invalid.")

        nodes = np.empty(self.n_ants, dtype=int)
        for ant_idx in range(self.n_ants):
            probs = weights[ant_idx] / row_sums[ant_idx]
            nodes[ant_idx] = self.rng.choice(self.n + 1, p=probs)
        return nodes

    def update_mask(self, travel_dis, cur_node, mask):
        mask[np.arange(self.n_ants), cur_node] = 0
        for ant_idx in range(self.n_ants):
            if cur_node[ant_idx] != self.n:
                candidates = np.flatnonzero(mask[ant_idx])
                trails = (
                    travel_dis[ant_idx]
                    + self.distances[cur_node[ant_idx], candidates]
                    + self.distances[candidates, 0]
                )
                mask[ant_idx, candidates[trails > self.max_len]] = 0

        mask[:, -1] = 0
        go_to_dummy = (mask[:, :-1] == 0).all(axis=1)
        mask[go_to_dummy, -1] = 1
        return mask

    @staticmethod
    def check_done(mask):
        return bool((mask[:, :-1] == 0).all())


def solve(coordinates: np.ndarray, heuristic: Callable, n_ants: int, n_iterations: int, rng: np.random.Generator) -> float:
    prize = prizes(coordinates)
    distance = distance_matrix(coordinates)
    maxlen = max_length(len(coordinates))

    heu = np.asarray(heuristic(prize.copy(), distance.copy(), maxlen), dtype=float)
    if heu.shape != distance.shape or not np.all(np.isfinite(heu)):
        return float("-inf")
    heu = heu + 1e-9
    heu[heu < 1e-9] = 1e-9

    aco = ACO(prize, distance, maxlen, heu, n_ants=n_ants, rng=rng)
    obj, _ = aco.run(n_iterations)
    return float(obj)


def _evaluate_op_aco_instance(heuristic: Callable, payload, context) -> float:
    idx, coordinates = payload
    rng = np.random.default_rng(None if context["seed"] is None else int(context["seed"]) + idx)
    try:
        return solve(
            coordinates=np.asarray(coordinates, dtype=float),
            heuristic=heuristic,
            n_ants=context["n_ants"],
            n_iterations=context["n_iterations"],
            rng=rng,
        )
    except Exception:
        return float("-inf")


class OPACOEvaluation(Evaluation):
    """Evaluator for OP ant colony optimization heuristic matrices."""

    def __init__(
            self,
            timeout_seconds=120,
            split: str = DEFAULT_SPLIT,
            n_ants: int = 20,
            n_iterations: int = 50,
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
            objs = evaluate_instances(
                program_str=program_str,
                callable_func=callable_func,
                payloads=list(enumerate(self._datasets)),
                instance_eval=_evaluate_op_aco_instance,
                context={
                    "n_ants": self.n_ants,
                    "n_iterations": self.n_iterations,
                    "seed": self.seed,
                },
                backend=self.eval_backend,
                workers=self.eval_workers,
                timeout_seconds=self.timeout_seconds,
            )
            return float(np.mean(objs))
        except Exception:
            return None

    def evaluate(self, heuristic: Callable) -> float:
        objs = []
        for idx, coordinates in enumerate(self._datasets):
            rng = np.random.default_rng(None if self.seed is None else int(self.seed) + idx)
            try:
                objs.append(solve(
                    coordinates=np.asarray(coordinates, dtype=float),
                    heuristic=heuristic,
                    n_ants=self.n_ants,
                    n_iterations=self.n_iterations,
                    rng=rng,
                ))
            except Exception:
                objs.append(float("-inf"))

        return float(np.mean(objs))

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.instance_parallel import evaluate_instances, validate_backend
from llm4ad.task.optimization.main.mkp_aco.dataset import (
    DEFAULT_SPLIT,
    load_split_instances,
)
from llm4ad.task.optimization.main.mkp_aco.template import template_program, task_description

__all__ = ["MKPACOEvaluation"]


class ACO:
    """MKP ant colony optimizer adapted from the MCTS-AHD benchmark."""

    def __init__(
            self,
            prize: np.ndarray,
            weight: np.ndarray,
            heuristic: np.ndarray,
            n_ants=10,
            decay=0.9,
            alpha=1,
            beta=1,
            rng: np.random.Generator | None = None,
    ):
        self.n, self.m = weight.shape
        self.prize = np.asarray(prize, dtype=float)
        self.weight = np.asarray(weight, dtype=float)
        self.heuristic = np.asarray(heuristic, dtype=float)
        self.n_ants = int(n_ants)
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.pheromone = np.ones(self.n + 1, dtype=float)
        self.Q = 1.0 / self.prize.sum()
        self.alltime_best_sol = None
        self.alltime_best_obj = 0.0
        self.rng = rng if rng is not None else np.random.default_rng()
        self.add_dummy_node()

    def add_dummy_node(self) -> None:
        self.prize = np.concatenate((self.prize, np.array([0.0])))
        self.weight = np.concatenate((self.weight, np.zeros((1, self.m))), axis=0)
        self.heuristic = np.concatenate((self.heuristic, np.array([1e-8])))

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
            self.pheromone[sol] += self.Q * obj

    def gen_sol_obj(self, solutions):
        return self.prize[solutions.T].sum(axis=1)

    def gen_sol(self):
        solutions = []
        knapsack = np.zeros((self.n_ants, self.m), dtype=float)
        mask = np.ones((self.n_ants, self.n + 1), dtype=float)
        dummy_mask = np.ones((self.n_ants, self.n + 1), dtype=float)
        dummy_mask[:, -1] = 0

        mask, knapsack = self.update_knapsack(mask, knapsack, new_item=None)
        dummy_mask = self.update_dummy_state(mask, dummy_mask)
        done = self.check_done(mask)
        while not done:
            items = self.pick_item(mask, dummy_mask)
            solutions.append(items)
            mask, knapsack = self.update_knapsack(mask, knapsack, items)
            dummy_mask = self.update_dummy_state(mask, dummy_mask)
            done = self.check_done(mask)

        return np.stack(solutions)

    def pick_item(self, mask, dummy_mask):
        pheromone = np.repeat(self.pheromone[np.newaxis, :], self.n_ants, axis=0)
        heuristic = np.repeat(self.heuristic[np.newaxis, :], self.n_ants, axis=0)
        weights = ((pheromone ** self.alpha) * (heuristic ** self.beta) * mask * dummy_mask)
        row_sums = weights.sum(axis=1)
        if np.any(row_sums <= 0) or not np.all(np.isfinite(row_sums)):
            raise ValueError("ACO transition probabilities are invalid.")

        items = np.empty(self.n_ants, dtype=int)
        for ant_idx in range(self.n_ants):
            probs = weights[ant_idx] / row_sums[ant_idx]
            items[ant_idx] = self.rng.choice(self.n + 1, p=probs)
        return items

    @staticmethod
    def check_done(mask):
        return bool((mask[:, :-1] == 0).all())

    @staticmethod
    def update_dummy_state(mask, dummy_mask):
        finished = (mask[:, :-1] == 0).all(axis=1)
        dummy_mask[finished] = 1
        return dummy_mask

    def update_knapsack(self, mask, knapsack, new_item):
        if new_item is not None:
            mask[np.arange(self.n_ants), new_item] = 0
            knapsack += self.weight[new_item]

        for ant_idx in range(self.n_ants):
            candidates = np.flatnonzero(mask[ant_idx])
            if len(candidates) > 1:
                new_knapsack = knapsack[ant_idx][np.newaxis, :] + self.weight[candidates]
                infeasible = candidates[(new_knapsack > 1).any(axis=1)]
                mask[ant_idx, infeasible] = 0
        mask[:, -1] = 1
        return mask, knapsack


def solve(prize: np.ndarray, weight: np.ndarray, heuristic: Callable, n_ants: int, n_iterations: int, rng) -> float:
    heu = np.asarray(heuristic(prize.copy(), weight.copy()), dtype=float)
    if heu.shape != (len(prize),) or not np.all(np.isfinite(heu)):
        return float("-inf")
    heu = heu + 1e-9
    heu[heu < 1e-9] = 1e-9

    aco = ACO(prize, weight, heu, n_ants=n_ants, rng=rng)
    obj, _ = aco.run(n_iterations)
    return float(obj)


def _evaluate_mkp_aco_instance(heuristic: Callable, payload, context) -> float:
    idx, prize, weight = payload
    rng = np.random.default_rng(None if context["seed"] is None else int(context["seed"]) + idx)
    try:
        return solve(
            prize=np.asarray(prize, dtype=float),
            weight=np.asarray(weight, dtype=float),
            heuristic=heuristic,
            n_ants=context["n_ants"],
            n_iterations=context["n_iterations"],
            rng=rng,
        )
    except Exception:
        return float("-inf")


class MKPACOEvaluation(Evaluation):
    """Evaluator for MKP ant colony optimization item heuristics."""

    def __init__(
            self,
            timeout_seconds=120,
            split: str = DEFAULT_SPLIT,
            n_ants: int = 10,
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
        self.n_items = int(self.dataset_metadata["n_items"])
        self.n_dimensions = int(self.dataset_metadata["n_dimensions"])
        self.n_ants = int(n_ants)
        self.n_iterations = int(n_iterations)
        self.seed = seed
        self.eval_workers = max(1, int(eval_workers))
        self.eval_backend = validate_backend(eval_backend, daemon_eval_process=self.daemon_eval_process)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        try:
            prizes = self._datasets["prizes"]
            weights = self._datasets["weights"]
            objs = evaluate_instances(
                program_str=program_str,
                callable_func=callable_func,
                payloads=[(idx, prize, weight) for idx, (prize, weight) in enumerate(zip(prizes, weights))],
                instance_eval=_evaluate_mkp_aco_instance,
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
        prizes = self._datasets["prizes"]
        weights = self._datasets["weights"]
        objs = []
        for idx, (prize, weight) in enumerate(zip(prizes, weights)):
            rng = np.random.default_rng(None if self.seed is None else int(self.seed) + idx)
            try:
                objs.append(solve(
                    prize=np.asarray(prize, dtype=float),
                    weight=np.asarray(weight, dtype=float),
                    heuristic=heuristic,
                    n_ants=self.n_ants,
                    n_iterations=self.n_iterations,
                    rng=rng,
                ))
            except Exception:
                objs.append(float("-inf"))

        return float(np.mean(objs))

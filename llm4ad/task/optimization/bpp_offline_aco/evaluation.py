from __future__ import annotations

from math import floor
from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.bpp_offline_aco.dataset import (
    CAPACITY,
    DEFAULT_SPLIT,
    load_split_instances,
)
from llm4ad.task.optimization.bpp_offline_aco.template import template_program, task_description

__all__ = ["BPPOfflineACOEvaluation"]


def organize_path(path: np.ndarray) -> tuple[int, np.ndarray]:
    order = {}
    result = np.zeros_like(path)
    for idx, value in enumerate(path):
        if value in order:
            result[idx] = order[value]
        else:
            result[idx] = order[value] = len(order)
    return len(order), result


def calculate_path_fitness(vacancies: list[int], capacity: int) -> float:
    occupied = capacity - np.array(vacancies, dtype=float)
    return float(((occupied / capacity) ** 2).sum().item() / len(vacancies))


class ACO:
    """Offline BPP ACO adapted from the MCTS-AHD benchmark."""

    def __init__(
            self,
            demand: np.ndarray,
            heuristic: np.ndarray,
            capacity: int,
            n_ants=20,
            decay=0.95,
            alpha=1,
            beta=1,
            greedy=False,
            rng: np.random.Generator | None = None,
    ):
        self.problem_size = len(demand)
        self.capacity = int(capacity)
        self.demand = np.asarray(demand, dtype=int)
        if self.demand.max() > self.capacity:
            raise ValueError("BPP demand exceeds bin capacity.")

        self.n_ants = int(n_ants)
        self.decay = float(decay)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.pheromone = np.ones((self.problem_size, self.problem_size), dtype=float)
        heuristic = np.asarray(heuristic, dtype=float).copy()
        heuristic[heuristic > 1e6] = 1e6
        heuristic[heuristic < 1e-6] = 1e-6
        heuristic = heuristic / heuristic.max()
        heuristic[heuristic < 1e-6] = 1e-6
        self.heuristic = heuristic

        self.shortest_path = np.arange(self.problem_size)
        self.best_cost = self.problem_size
        self._ordinal = np.arange(self.problem_size, dtype=int)
        self.greedy_mode = bool(greedy)

    def run(self, iterations: int) -> tuple[int, np.ndarray]:
        for _ in range(int(iterations)):
            prob = self.pheromone ** self.alpha * self.heuristic ** self.beta
            paths, costs, fitnesses = self.gen_paths(self.n_ants, prob)
            best_index = int(costs.argmin())
            best_cost = int(costs[best_index].item())
            if best_cost < self.best_cost:
                self.shortest_path = paths[best_index]
                self.best_cost = best_cost
            self.update_pheronome(paths, fitnesses)
        return organize_path(self.shortest_path)

    def sample_only(self, count: int) -> tuple[int, np.ndarray]:
        self.greedy_mode = True
        paths, costs, _ = self.gen_paths(count, self.heuristic)
        best_index = int(costs.argmin())
        return organize_path(paths[best_index])

    def update_pheronome(self, paths: list[np.ndarray], fitnesses: np.ndarray) -> None:
        delta_phe = np.zeros_like(self.pheromone)
        for path, fitness in zip(paths, fitnesses):
            delta_phe[path[:, None] == path[None, :]] += fitness / self.n_ants
        self.pheromone *= self.decay
        self.pheromone += delta_phe

    def gen_paths(self, count: int, prob: np.ndarray) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        paths, costs, fitnesses = [], [], []
        for _ in range(int(count)):
            path, cost, fitness = self.sample_path(prob)
            paths.append(path)
            costs.append(cost)
            fitnesses.append(fitness)
        return paths, np.array(costs, dtype=int), np.array(fitnesses, dtype=float)

    def sample_path(self, prob: np.ndarray) -> tuple[np.ndarray, int, float]:
        path = np.ones(self.problem_size, dtype=int) * -1
        valid_items = np.ones(self.problem_size, dtype=bool)
        current_bin = item_count = 0
        vacancies: list[int] = []
        bin_vacancy = self.capacity
        bin_items = np.zeros_like(valid_items)

        for _ in range(self.problem_size):
            mask = np.bitwise_and(self.demand <= bin_vacancy, valid_items)
            if not np.any(mask):
                vacancies.append(bin_vacancy)
                bin_vacancy, item_count = self.capacity, 0
                current_bin += 1
                bin_items[:] = False
                selected = self.random_select(valid_items)
            else:
                if item_count == 0:
                    selected = self.random_select(mask)
                else:
                    item_prob = (prob[bin_items].sum(0) / item_count + 1e-5) * mask
                    selected = self.greedy_sample(item_prob) if self.greedy_mode else self.random_sample(item_prob)

            bin_items[selected] = True
            bin_vacancy -= int(self.demand[selected])
            valid_items[selected] = False
            path[selected] = current_bin
            item_count += 1

        vacancies.append(bin_vacancy)
        fitness = calculate_path_fitness(vacancies, self.capacity)
        return path, len(vacancies), fitness

    def random_select(self, mask: np.ndarray) -> int:
        valid = self._ordinal[mask]
        return int(valid[floor(self.rng.random() * len(valid))].item())

    def random_sample(self, prob: np.ndarray) -> int:
        cumprob = np.cumsum(prob)
        sampled = int(np.searchsorted(cumprob, self.rng.random() * cumprob[-1]).item())
        return sampled if sampled < len(cumprob) else len(cumprob) - 1

    @staticmethod
    def greedy_sample(prob: np.ndarray) -> int:
        return int(prob.argmax().item())


def solve(
        demand: np.ndarray,
        heuristic: Callable,
        n_ants: int,
        n_iterations: int,
        sample_count: int,
        mode: str,
        rng: np.random.Generator,
) -> float:
    heu = np.asarray(heuristic(demand.copy(), CAPACITY), dtype=float)
    if heu.shape != (len(demand), len(demand)) or not np.all(np.isfinite(heu)):
        return float("inf")
    if heu.max() <= 0:
        return float("inf")

    aco = ACO(demand=demand, heuristic=heu, capacity=CAPACITY, n_ants=n_ants, greedy=False, rng=rng)
    if mode == "sample":
        cost, _ = aco.sample_only(sample_count)
    else:
        cost, _ = aco.run(n_iterations)
    return float(cost)


class BPPOfflineACOEvaluation(Evaluation):
    """Evaluator for offline BPP ant colony optimization heuristic matrices."""

    def __init__(
            self,
            timeout_seconds=120,
            split: str = DEFAULT_SPLIT,
            n_ants: int = 20,
            n_iterations: int = 15,
            sample_count: int = 200,
            mode: str = "aco",
            seed: int | None = 1234,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )
        if mode not in {"aco", "sample"}:
            raise ValueError("mode must be `aco` or `sample`.")

        self._datasets, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.n_items = int(self.dataset_metadata["n_items"])
        self.capacity = int(self.dataset_metadata["capacity"])
        self.n_ants = int(n_ants)
        self.n_iterations = int(n_iterations)
        self.sample_count = int(sample_count)
        self.mode = mode
        self.seed = seed

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, heuristic: Callable) -> float:
        costs = []
        for idx, demand in enumerate(self._datasets):
            rng = np.random.default_rng(None if self.seed is None else int(self.seed) + idx)
            try:
                costs.append(solve(
                    demand=np.asarray(demand, dtype=int),
                    heuristic=heuristic,
                    n_ants=self.n_ants,
                    n_iterations=self.n_iterations,
                    sample_count=self.sample_count,
                    mode=self.mode,
                    rng=rng,
                ))
            except Exception:
                costs.append(float("inf"))

        return -float(np.mean(costs))

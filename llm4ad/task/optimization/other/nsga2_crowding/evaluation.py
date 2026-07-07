from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.other.nsga2_crowding.dataset import load_split_instances
from llm4ad.task.optimization.other.nsga2_crowding.template import task_description, template_program

__all__ = ["NSGA2CrowdingEvaluation"]


def non_dominated_sort(F: np.ndarray) -> list[list[int]]:
    n = len(F)
    dominates = (
        np.all(F[:, np.newaxis] <= F[np.newaxis], axis=2)
        & np.any(F[:, np.newaxis] < F[np.newaxis], axis=2)
    )
    n_dominated_by = dominates.sum(axis=0).astype(int)

    fronts = []
    assigned = np.zeros(n, dtype=bool)
    while not np.all(assigned):
        mask = ~assigned & (n_dominated_by == 0)
        if not np.any(mask):
            fronts.append(np.where(~assigned)[0].tolist())
            break
        front = np.where(mask)[0]
        fronts.append(front.tolist())
        assigned[front] = True
        n_dominated_by -= dominates[np.ix_(front, np.arange(n))].sum(axis=0)
    return fronts


def sbx(rng: np.random.Generator, x1: np.ndarray, x2: np.ndarray, eta: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    c1, c2 = x1.copy(), x2.copy()
    for i in range(len(x1)):
        if rng.random() < 0.5 and abs(x1[i] - x2[i]) > 1e-10:
            u = rng.random()
            beta = (
                (2 * u) ** (1.0 / (eta + 1))
                if u <= 0.5
                else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1))
            )
            c1[i] = np.clip(0.5 * ((x1[i] + x2[i]) - beta * abs(x2[i] - x1[i])), 0.0, 1.0)
            c2[i] = np.clip(0.5 * ((x1[i] + x2[i]) + beta * abs(x2[i] - x1[i])), 0.0, 1.0)
    return c1, c2


def polynomial_mutation(rng: np.random.Generator, x: np.ndarray, eta: float = 20.0) -> np.ndarray:
    y = x.copy()
    for i in range(len(y)):
        if rng.random() < 1.0 / len(y):
            u = rng.random()
            delta = (
                (2 * u) ** (1.0 / (eta + 1)) - 1.0
                if u < 0.5
                else 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1))
            )
            y[i] = np.clip(y[i] + delta, 0.0, 1.0)
    return y


def hypervolume_2d(F: np.ndarray, ref_point: np.ndarray) -> float:
    feasible = F[np.all(F < ref_point, axis=1)]
    if len(feasible) == 0:
        return 0.0
    sorted_front = feasible[np.argsort(feasible[:, 0])]
    nondominated = [sorted_front[0]]
    for i in range(1, len(sorted_front)):
        if sorted_front[i, 1] < nondominated[-1][1]:
            nondominated.append(sorted_front[i])
    nondominated = np.array(nondominated)
    hv = 0.0
    for i in range(len(nondominated)):
        f1_next = nondominated[i + 1, 0] if i + 1 < len(nondominated) else ref_point[0]
        hv += (f1_next - nondominated[i, 0]) * (ref_point[1] - nondominated[i, 1])
    return float(hv)


def pareto_front(F: np.ndarray) -> np.ndarray:
    n = len(F)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        for j in range(n):
            if i != j and not is_dominated[j]:
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
    return F[~is_dominated]


class NSGA2CrowdingEvaluation(Evaluation):
    """Evaluator for EoH's NSGA-II crowding-distance design task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            pop_size: int | None = None,
            n_gen: int | None = None,
            n_runs: int | None = None,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._instances, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.pop_size = int(pop_size if pop_size is not None else self.dataset_metadata["pop_size"])
        self.n_gen = int(n_gen if n_gen is not None else self.dataset_metadata["n_gen"])
        self.n_runs = int(n_runs if n_runs is not None else self.dataset_metadata["n_runs"])
        self.seed_start = int(self.dataset_metadata["seed_start"])

    def _crowding_scores(self, F: np.ndarray, crowding_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        scores = np.asarray(crowding_fn(F), dtype=float).ravel()
        if scores.shape != (len(F),):
            raise ValueError("crowding_distance returned an invalid shape")
        return scores

    def _run_nsga2(
            self,
            instance: dict[str, Any],
            crowding_fn: Callable[[np.ndarray], np.ndarray],
            seed: int,
    ) -> float:
        rng = np.random.default_rng(seed)
        problem_fn = instance["func"]
        n_var = int(instance["n_var"])
        ref_point = instance["ref_point"]
        pop_size = self.pop_size

        X = rng.uniform(0.0, 1.0, (pop_size, n_var))
        F = np.array([problem_fn(x) for x in X])

        fronts = non_dominated_sort(F)
        rank = np.zeros(pop_size, dtype=int)
        crowding = np.zeros(pop_size)
        for front_rank, front in enumerate(fronts):
            rank[front] = front_rank
            scores = self._crowding_scores(F[front], crowding_fn) if len(front) > 1 else np.full(len(front), np.inf)
            crowding[front] = scores

        for _ in range(self.n_gen):
            offspring_X, offspring_F = [], []
            while len(offspring_X) < pop_size:
                a, b = rng.integers(0, pop_size), rng.integers(0, pop_size)
                p1 = a if rank[a] < rank[b] or (rank[a] == rank[b] and crowding[a] >= crowding[b]) else b
                a, b = rng.integers(0, pop_size), rng.integers(0, pop_size)
                p2 = a if rank[a] < rank[b] or (rank[a] == rank[b] and crowding[a] >= crowding[b]) else b

                c1, c2 = sbx(rng, X[p1], X[p2])
                c1 = polynomial_mutation(rng, c1)
                c2 = polynomial_mutation(rng, c2)
                offspring_X.extend([c1, c2])
                offspring_F.extend([problem_fn(c1), problem_fn(c2)])

            QX = np.array(offspring_X[:pop_size])
            QF = np.array(offspring_F[:pop_size])
            RX = np.vstack([X, QX])
            RF = np.vstack([F, QF])
            fronts = non_dominated_sort(RF)

            crowding_r = np.zeros(2 * pop_size)
            for front in fronts:
                scores = self._crowding_scores(RF[front], crowding_fn) if len(front) > 1 else np.full(len(front), np.inf)
                crowding_r[front] = scores

            new_X, new_F = [], []
            for front in fronts:
                if len(new_X) + len(front) <= pop_size:
                    new_X.extend(RX[front].tolist())
                    new_F.extend(RF[front].tolist())
                else:
                    remaining = pop_size - len(new_X)
                    best = sorted(front, key=lambda idx: -crowding_r[idx])[:remaining]
                    new_X.extend(RX[best].tolist())
                    new_F.extend(RF[best].tolist())
                    break

            X = np.array(new_X)
            F = np.array(new_F)
            fronts = non_dominated_sort(F)
            rank = np.zeros(pop_size, dtype=int)
            crowding = np.zeros(pop_size)
            for front_rank, front in enumerate(fronts):
                rank[front] = front_rank
                scores = self._crowding_scores(F[front], crowding_fn) if len(front) > 1 else np.full(len(front), np.inf)
                crowding[front] = scores

        return hypervolume_2d(pareto_front(F), ref_point)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, crowding_fn: Callable[[np.ndarray], np.ndarray]) -> float | None:
        try:
            hypervolumes = []
            for instance in self._instances:
                runs = [
                    self._run_nsga2(instance, crowding_fn, seed)
                    for seed in range(self.seed_start, self.seed_start + self.n_runs)
                ]
                hypervolumes.append(float(np.mean(runs)))
            return float(np.mean(hypervolumes))
        except Exception:
            return None

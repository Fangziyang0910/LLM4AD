from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.other.moead_decomposition.dataset import load_split_instances
from llm4ad.task.optimization.other.moead_decomposition.template import task_description, template_program

__all__ = ["MOEADDecompositionEvaluation"]


class MOEADDecompositionEvaluation(Evaluation):
    """Evaluator for EoH's MOEA/D decomposition-operator task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            n_gen: int | None = None,
            n_runs: int | None = None,
            T: int | None = None,
            hv_samples: int | None = None,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._instances, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.n_gen = int(n_gen if n_gen is not None else self.dataset_metadata["n_gen"])
        self.n_runs = int(n_runs if n_runs is not None else self.dataset_metadata["n_runs"])
        self.seed_start = int(self.dataset_metadata["seed_start"])
        self.T = int(T if T is not None else self.dataset_metadata["T"])
        self.H = int(self.dataset_metadata["H"])
        self.hv_samples = int(
            hv_samples if hv_samples is not None else self.dataset_metadata["hv_samples"]
        )
        self._weights_cache: dict[tuple[int, int], np.ndarray] = {}

    def _das_dennis_weights(self, n_obj: int, H: int) -> np.ndarray:
        key = (n_obj, H)
        if key in self._weights_cache:
            return self._weights_cache[key]

        weights: list[list[float]] = []

        def recurse(remaining: int, n_left: int, current: list[float]) -> None:
            if n_left == 1:
                weights.append(current + [remaining / H])
                return
            for i in range(remaining + 1):
                recurse(remaining - i, n_left - 1, current + [i / H])

        recurse(H, n_obj, [])
        result = np.array(weights, dtype=float)
        result = np.where(result == 0.0, 1e-6, result)
        self._weights_cache[key] = result
        return result

    @staticmethod
    def _get_pareto_front(F: np.ndarray) -> np.ndarray:
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return F[~is_dominated]

    def _hypervolume_mc(self, F: np.ndarray, ref_point: np.ndarray) -> float:
        feasible = F[np.all(F < ref_point, axis=1)]
        if len(feasible) == 0:
            return 0.0
        rng = np.random.default_rng(0)
        samples = rng.uniform(0.0, ref_point, size=(self.hv_samples, len(ref_point)))
        dominated = np.any(
            np.all(feasible[:, np.newaxis, :] <= samples[np.newaxis, :, :], axis=2),
            axis=0,
        )
        return float(np.mean(dominated) * np.prod(ref_point))

    def _run_moead(
            self,
            instance: dict[str, Any],
            decomp_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
            seed: int,
    ) -> float:
        rng = np.random.default_rng(seed)
        problem_fn = instance["func"]
        n_var = int(instance["n_var"])
        n_obj = int(instance["n_obj"])
        ref_point = instance["ref_point"]

        weights = self._das_dennis_weights(n_obj, self.H)
        pop_size = len(weights)

        distances = np.sum((weights[:, np.newaxis, :] - weights[np.newaxis, :, :]) ** 2, axis=2)
        neighbors = np.argsort(distances, axis=1)[:, : self.T]

        X = rng.uniform(0.0, 1.0, (pop_size, n_var))
        F_vals = np.array([problem_fn(x) for x in X])
        ideal_point = np.min(F_vals, axis=0).copy()

        for _ in range(self.n_gen):
            for i in rng.permutation(pop_size):
                nb = neighbors[i]
                p1_idx, p2_idx = rng.choice(nb, 2, replace=False)
                r_idx = rng.integers(0, pop_size)
                mutant = X[p1_idx] + 0.5 * (X[p2_idx] - X[r_idx])
                cross_mask = rng.random(n_var) < 0.9
                cross_mask[rng.integers(n_var)] = True
                child = np.where(cross_mask, mutant, X[i])
                child = np.clip(child, 0.0, 1.0)

                child_F = problem_fn(child)
                ideal_point = np.minimum(ideal_point, child_F)

                F_nb = F_vals[nb]
                W_nb = weights[nb]
                child_F_batch = np.tile(child_F, (len(nb), 1))

                old_scores = np.asarray(decomp_fn(F_nb, W_nb, ideal_point), dtype=float).ravel()
                new_scores = np.asarray(decomp_fn(child_F_batch, W_nb, ideal_point), dtype=float).ravel()
                if old_scores.shape != (len(nb),) or new_scores.shape != (len(nb),):
                    raise ValueError("custom_decomposition returned an invalid score shape")
                if not (np.all(np.isfinite(old_scores)) and np.all(np.isfinite(new_scores))):
                    raise ValueError("custom_decomposition returned non-finite scores")

                update_mask = new_scores <= old_scores
                X[nb[update_mask]] = child
                F_vals[nb[update_mask]] = child_F

        pareto_F = self._get_pareto_front(F_vals)
        return self._hypervolume_mc(pareto_F, ref_point)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, decomp_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> float | None:
        try:
            hypervolumes = []
            for instance in self._instances:
                runs = [
                    self._run_moead(instance, decomp_fn, seed)
                    for seed in range(self.seed_start, self.seed_start + self.n_runs)
                ]
                hypervolumes.append(float(np.mean(runs)))
            return float(np.mean(hypervolumes))
        except Exception:
            return None

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.cmaes_cov_update.dataset import load_split_instances
from llm4ad.task.optimization.cmaes_cov_update.template import task_description, template_program
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT

__all__ = ["CMAESCovUpdateEvaluation"]


class CMAESCovUpdateEvaluation(Evaluation):
    """Evaluator for EoH's CMA-ES covariance-update task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            max_evals: int | None = None,
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
        self.max_evals = int(max_evals if max_evals is not None else self.dataset_metadata["max_evals"])
        self.n_runs = int(n_runs if n_runs is not None else self.dataset_metadata["n_runs"])
        self.seed_start = int(self.dataset_metadata["seed_start"])

    def _run_cmaes(
            self,
            instance: dict[str, Any],
            update_cov_fn: Callable[..., np.ndarray],
    ) -> float:
        func = instance["func"]
        n = int(instance["dim"])
        lo, hi = instance["bounds"]

        lam = 4 + int(3 * np.log(n))
        mu = lam // 2

        weights_raw = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
        weights = weights_raw / weights_raw.sum()
        mueff = weights.sum() ** 2 / (weights ** 2).sum()

        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        mean = lo + (hi - lo) * np.random.rand(n)
        sigma = (hi - lo) / 3.0
        covariance = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        best_f = np.inf
        n_evals = 0
        generation = 0

        while n_evals < self.max_evals:
            eigvals, basis = np.linalg.eigh(covariance)
            eigvals = np.maximum(eigvals, 1e-20)
            scales = np.sqrt(eigvals)
            invsqrt_covariance = (basis / scales) @ basis.T

            z = np.random.randn(lam, n)
            y = z * scales @ basis.T
            x = np.clip(mean + sigma * y, lo, hi)

            f_vals = np.array([func(xi) for xi in x])
            n_evals += lam
            generation += 1

            idx = np.argsort(f_vals)
            best_f = min(best_f, float(f_vals[idx[0]]))

            y_sel = y[idx[:mu]]
            mean_old = mean.copy()
            mean = weights @ y_sel * sigma + mean_old
            y_w = weights @ y_sel

            p_sigma = (
                (1 - cs) * p_sigma
                + np.sqrt(cs * (2 - cs) * mueff) * (invsqrt_covariance @ y_w)
            )
            norm_ps = float(np.linalg.norm(p_sigma))

            threshold = (1.4 + 2 / (n + 1)) * chi_n
            norm_ps_adjusted = norm_ps / np.sqrt(1 - (1 - cs) ** (2 * generation))
            hsig = 1.0 if norm_ps_adjusted < threshold else 0.0

            p_c = (
                (1 - cc) * p_c
                + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w
            )

            new_covariance = update_cov_fn(
                covariance.copy(),
                p_c.copy(),
                weights.copy(),
                y_sel.copy(),
                float(c1),
                float(cmu),
                float(cc),
                hsig,
                n,
            )
            new_covariance = np.asarray(new_covariance, dtype=float)
            if new_covariance.shape != (n, n):
                raise ValueError(
                    f"update_covariance returned shape {new_covariance.shape}, expected ({n}, {n})."
                )
            if not np.all(np.isfinite(new_covariance)):
                raise ValueError("update_covariance returned non-finite values.")
            covariance = (new_covariance + new_covariance.T) / 2.0

            sigma *= float(np.exp((cs / damps) * (norm_ps / chi_n - 1)))
            sigma = float(np.clip(sigma, 1e-12, 1e6))

        return float(best_f)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, update_cov_fn: Callable[..., np.ndarray]) -> float | None:
        try:
            scores = []
            for instance in self._instances:
                run_bests = []
                for seed in range(self.seed_start, self.seed_start + self.n_runs):
                    np.random.seed(seed)
                    run_bests.append(self._run_cmaes(instance, update_cov_fn))
                scores.append(float(np.log1p(np.mean(run_bests))))
            return -float(np.mean(scores))
        except Exception:
            return None

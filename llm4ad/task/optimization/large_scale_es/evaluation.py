from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.large_scale_es.dataset import load_split_instances
from llm4ad.task.optimization.large_scale_es.template import task_description, template_program

__all__ = ["LargeScaleESEvaluation"]


class LargeScaleESEvaluation(Evaluation):
    """Evaluator for EoH's large-scale separable CMA-ES task."""

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

    @staticmethod
    def _sepcmaes_params(n: int) -> dict[str, Any]:
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        raw_weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1, dtype=float))
        weights = raw_weights / raw_weights.sum()
        mueff = float(1.0 / np.sum(weights ** 2))
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        c1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))
        return {
            "lam": lam,
            "mu": mu,
            "weights": weights,
            "mueff": mueff,
            "cs": cs,
            "damps": damps,
            "cc": cc,
            "c1": c1,
            "cmu": cmu,
            "chi_n": chi_n,
        }

    def _run_sepcmaes(
            self,
            instance: dict[str, Any],
            adapt_fn: Callable[..., np.ndarray],
    ) -> float:
        func = instance["func"]
        n = int(instance["dim"])
        lo, hi = instance["bounds"]
        domain = hi - lo
        params = self._sepcmaes_params(n)
        lam = params["lam"]
        mu = params["mu"]
        weights = params["weights"]
        mueff = params["mueff"]
        cs = params["cs"]
        damps = params["damps"]
        cc = params["cc"]
        c1 = params["c1"]
        cmu = params["cmu"]
        chi_n = params["chi_n"]
        max_generations = (self.max_evals - 1) // lam

        mean = lo + domain * np.random.rand(n)
        sigma = domain / 4.0
        diagonal_cov = np.ones(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        best_f = np.inf
        n_evals = 0
        generation = 0

        while n_evals < self.max_evals:
            z = np.random.randn(lam, n)
            y = z * np.sqrt(diagonal_cov)
            x = np.clip(mean + sigma * y, lo, hi)

            f_vals = np.array([func(xi) for xi in x])
            n_evals += lam

            idx = np.argsort(f_vals)
            best_f = min(best_f, float(f_vals[idx[0]]))
            y_sel = y[idx[:mu]]
            y_w = weights @ y_sel

            mean = np.clip(mean + sigma * y_w, lo, hi)

            inv_sqrt_d = 1.0 / np.sqrt(np.maximum(diagonal_cov, 1e-20))
            p_sigma = (
                (1.0 - cs) * p_sigma
                + np.sqrt(cs * (2.0 - cs) * mueff) * inv_sqrt_d * y_w
            )
            norm_ps = float(np.linalg.norm(p_sigma))

            adjusted = norm_ps / np.sqrt(
                max(1e-20, 1.0 - (1.0 - cs) ** (2.0 * (generation + 1)))
            )
            hsig = 1.0 if adjusted < (1.4 + 2.0 / (n + 1.0)) * chi_n else 0.0

            p_c = (
                (1.0 - cc) * p_c
                + hsig * np.sqrt(cc * (2.0 - cc) * mueff) * y_w
            )

            new_diagonal_cov = adapt_fn(
                diagonal_cov.copy(),
                p_c.copy(),
                weights.copy(),
                y_sel.copy(),
                float(c1),
                float(cmu),
                float(cc),
                float(hsig),
                int(n),
                int(generation),
                int(max_generations),
            )
            new_diagonal_cov = np.asarray(new_diagonal_cov, dtype=float)
            if new_diagonal_cov.shape != (n,):
                raise ValueError(
                    f"adapt_diagonal_cov returned shape {new_diagonal_cov.shape}, expected ({n},)."
                )
            if not np.all(np.isfinite(new_diagonal_cov)):
                raise ValueError("adapt_diagonal_cov returned non-finite values.")
            diagonal_cov = np.clip(new_diagonal_cov, 1e-20, 1e10)

            sigma *= float(np.exp((cs / damps) * (norm_ps / chi_n - 1.0)))
            sigma = float(np.clip(sigma, 1e-12, domain))
            generation += 1

        return float(best_f)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, adapt_fn: Callable[..., np.ndarray]) -> float | None:
        try:
            scores = []
            for instance in self._instances:
                run_bests = []
                for seed in range(self.seed_start, self.seed_start + self.n_runs):
                    np.random.seed(seed)
                    run_bests.append(self._run_sepcmaes(instance, adapt_fn))
                scores.append(float(np.log1p(np.mean(run_bests))))
            return -float(np.mean(scores))
        except Exception:
            return None

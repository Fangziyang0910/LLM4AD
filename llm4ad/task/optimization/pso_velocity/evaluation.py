from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.pso_velocity.dataset import load_split_instances
from llm4ad.task.optimization.pso_velocity.template import task_description, template_program

__all__ = ["PSOVelocityEvaluation"]


class PSOVelocityEvaluation(Evaluation):
    """Evaluator for EoH's PSO velocity-update task."""

    def __init__(
            self,
            timeout_seconds=60,
            split: str = DEFAULT_SPLIT,
            pop_size: int | None = None,
            max_iterations: int | None = None,
            n_runs: int | None = None,
            w: float | None = None,
            c1: float | None = None,
            c2: float | None = None,
            v_max_ratio: float | None = None,
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
        self.max_iterations = int(
            max_iterations if max_iterations is not None else self.dataset_metadata["max_iterations"]
        )
        self.n_runs = int(n_runs if n_runs is not None else self.dataset_metadata["n_runs"])
        self.w = float(w if w is not None else self.dataset_metadata["w"])
        self.c1 = float(c1 if c1 is not None else self.dataset_metadata["c1"])
        self.c2 = float(c2 if c2 is not None else self.dataset_metadata["c2"])
        self.v_max_ratio = float(
            v_max_ratio if v_max_ratio is not None else self.dataset_metadata["v_max_ratio"]
        )
        self.seed_start = int(self.dataset_metadata["seed_start"])

    def _run_pso(
            self,
            instance: dict[str, Any],
            update_velocity_fn: Callable[..., np.ndarray],
    ) -> float:
        func = instance["func"]
        dim = int(instance["dim"])
        lo, hi = instance["bounds"]
        bounds = np.column_stack([np.full(dim, lo), np.full(dim, hi)])
        v_max = self.v_max_ratio * (hi - lo)

        positions = lo + (hi - lo) * np.random.rand(self.pop_size, dim)
        velocities = -v_max + 2 * v_max * np.random.rand(self.pop_size, dim)
        fitness = np.array([func(p) for p in positions])

        pbest_positions = positions.copy()
        pbest_fitness = fitness.copy()
        gbest_idx = int(np.argmin(pbest_fitness))
        gbest_position = pbest_positions[gbest_idx].copy()
        gbest_fitness = float(pbest_fitness[gbest_idx])

        for iteration in range(self.max_iterations):
            new_velocities = update_velocity_fn(
                velocities.copy(),
                positions.copy(),
                pbest_positions.copy(),
                pbest_fitness.copy(),
                gbest_position.copy(),
                gbest_fitness,
                self.w,
                self.c1,
                self.c2,
                bounds,
                iteration,
                self.max_iterations,
            )
            new_velocities = np.asarray(new_velocities, dtype=float)
            if new_velocities.shape != (self.pop_size, dim):
                raise ValueError(
                    f"update_velocity returned shape {new_velocities.shape}, "
                    f"expected ({self.pop_size}, {dim})."
                )
            if not np.all(np.isfinite(new_velocities)):
                raise ValueError("update_velocity returned non-finite values.")

            velocities = np.clip(new_velocities, -v_max, v_max)
            positions = np.clip(positions + velocities, lo, hi)

            fitness = np.array([func(p) for p in positions])
            improved = fitness < pbest_fitness
            pbest_positions[improved] = positions[improved].copy()
            pbest_fitness[improved] = fitness[improved]

            best_idx = int(np.argmin(pbest_fitness))
            if pbest_fitness[best_idx] < gbest_fitness:
                gbest_fitness = float(pbest_fitness[best_idx])
                gbest_position = pbest_positions[best_idx].copy()

        return gbest_fitness

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, update_velocity_fn: Callable[..., np.ndarray]) -> float | None:
        try:
            scores = []
            for instance in self._instances:
                run_bests = []
                for seed in range(self.seed_start, self.seed_start + self.n_runs):
                    np.random.seed(seed)
                    run_bests.append(self._run_pso(instance, update_velocity_fn))
                scores.append(float(np.log1p(np.mean(run_bests))))
            return -float(np.mean(scores))
        except Exception:
            return None

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.instance_parallel import evaluate_instances, validate_backend
from llm4ad.task.optimization.main.dpp_ga.dataset import (
    DEFAULT_SPLIT,
    load_problem_arrays,
    load_split_instances,
)
from llm4ad.task.optimization.main.dpp_ga.template import task_description, template_program

__all__ = ["DPPGAEvaluation", "seed_crossover"]


def seed_crossover(parents: np.ndarray, n_pop: int) -> np.ndarray:
    n_parents, n_decap = parents.shape
    left_halves = parents[:, :n_decap // 2]
    right_halves = parents[:, n_decap // 2:]
    parent_pairs = np.stack([
        np.random.choice(range(n_parents), 2, replace=False)
        for _ in range(n_pop)
    ])
    return np.concatenate([
        left_halves[parent_pairs[:, 0]],
        right_halves[parent_pairs[:, 1]],
    ], axis=1)


def _initial_impedance(raw_pdn: np.ndarray, probe: int) -> np.ndarray:
    return raw_pdn[:, int(probe), int(probe)]


def _decap_placement(
        *,
        raw_pdn: np.ndarray,
        decap: np.ndarray,
        pi: np.ndarray,
        probe: int,
        n_ports: int,
        freq_pts: int,
) -> np.ndarray:
    pi = np.asarray(pi, dtype=int)
    n_decap = int(np.size(pi))
    z2 = np.zeros((freq_pts, n_decap, n_decap), dtype=raw_pdn.dtype)
    q_idx = np.arange(n_decap)
    z2[:, q_idx, q_idx] = np.abs(decap)[:, None]

    all_idx = np.arange(n_ports)
    a_idx = np.delete(all_idx, pi)

    z1aa = raw_pdn[:, a_idx, :][:, :, a_idx]
    z1ap = raw_pdn[:, a_idx, :][:, :, pi]
    z1pa = raw_pdn[:, pi, :][:, :, a_idx]
    z1pp = raw_pdn[:, pi, :][:, :, pi]
    z2qq = z2[:, q_idx, :][:, :, q_idx]

    zout = z1aa - np.matmul(np.matmul(z1ap, np.linalg.inv(z1pp + z2qq)), z1pa)
    adjusted_probe = int(probe) - int(np.sum(pi < int(probe)))
    return zout[:, adjusted_probe, adjusted_probe]


def _reward_model(
        *,
        arrays: dict[str, np.ndarray],
        probe: int,
        pi: np.ndarray,
        n: int,
        m: int,
        freq_pts: int,
) -> float:
    z_initial = np.abs(_initial_impedance(arrays["raw_pdn"], probe))
    z_final = np.abs(_decap_placement(
        raw_pdn=arrays["raw_pdn"],
        decap=arrays["decap"],
        pi=pi,
        probe=probe,
        n_ports=n * m,
        freq_pts=freq_pts,
    ))
    impedance_gap = z_initial - z_final
    return float(np.sum(impedance_gap * 1000000000 / arrays["freq"]) / 10)


def _generate_population(
        *,
        n_pop: int,
        n_decap: int,
        probe: int,
        prohibit: np.ndarray,
        size: int,
) -> np.ndarray:
    possible = np.setdiff1d(np.arange(size), np.append(prohibit, probe))
    if len(possible) < n_decap:
        raise ValueError("Not enough feasible actions for DPP-GA population.")
    return np.stack([
        np.random.choice(possible, n_decap, replace=False)
        for _ in range(n_pop)
    ])


def _validate_population(
        population: np.ndarray,
        *,
        probe: int,
        prohibit: np.ndarray,
        size: int,
) -> np.ndarray:
    population = np.asarray(population, dtype=int).copy()
    n_pop, n_decap = population.shape
    for i in range(n_pop):
        ind = population[i]
        unique_actions = np.unique(ind)
        if len(unique_actions) == n_decap:
            continue

        duplicate_indices = []
        action_set = set()
        for j, action in enumerate(ind):
            if int(action) in action_set:
                duplicate_indices.append(j)
            action_set.add(int(action))

        infeasible = np.concatenate([prohibit, [probe], unique_actions])
        feasible = np.setdiff1d(np.arange(size), infeasible)
        if len(feasible) < len(duplicate_indices):
            raise ValueError("Not enough feasible actions to repair DPP-GA offspring.")
        ind[duplicate_indices] = np.random.choice(feasible, len(duplicate_indices), replace=False)
    return population


def _eval_population(
        population: np.ndarray,
        *,
        probe: int,
        arrays: dict[str, np.ndarray],
        n: int,
        m: int,
        freq_pts: int,
) -> np.ndarray:
    return np.array([
        _reward_model(
            arrays=arrays,
            probe=probe,
            pi=individual,
            n=n,
            m=m,
            freq_pts=freq_pts,
        )
        for individual in population
    ], dtype=float)


def _selection(population: np.ndarray, rewards: np.ndarray) -> np.ndarray:
    return population[int(len(population) / 2):]


def _run_dpp_ga_instance(crossover: Callable, instance: dict[str, Any], context: dict[str, Any]) -> float:
    probe = int(instance["probe"])
    keepout = np.asarray(instance["keepout"], dtype=int)
    prohibit = keepout[:int(instance["keepout_num"])]
    size = context["n"] * context["m"]

    population = _generate_population(
        n_pop=context["n_pop"],
        n_decap=context["n_decap"],
        probe=probe,
        prohibit=prohibit,
        size=size,
    )
    rewards = _eval_population(
        population,
        probe=probe,
        arrays=context["arrays"],
        n=context["n"],
        m=context["m"],
        freq_pts=context["freq_pts"],
    )

    n_elite = max(1, int(context["n_pop"] * context["elite_rate"]))
    for _ in range(context["n_iter"]):
        sorted_idx = rewards.argsort()
        population = population[sorted_idx]
        rewards = rewards[sorted_idx]
        selected_population = _selection(population, rewards)
        elites = population[-n_elite:]
        elite_rewards = rewards[-n_elite:]

        next_population = np.asarray(
            crossover(selected_population, n_pop=context["n_pop"] - n_elite),
            dtype=int,
        )
        if next_population.shape != (context["n_pop"] - n_elite, context["n_decap"]):
            raise ValueError("DPP-GA crossover returned an invalid offspring shape.")
        next_population = _validate_population(
            next_population,
            probe=probe,
            prohibit=prohibit,
            size=size,
        )
        next_rewards = _eval_population(
            next_population,
            probe=probe,
            arrays=context["arrays"],
            n=context["n"],
            m=context["m"],
            freq_pts=context["freq_pts"],
        )
        population = np.concatenate([elites, next_population], axis=0)
        rewards = np.concatenate([elite_rewards, next_rewards], axis=0)

    return float(np.max(rewards))


def _evaluate_dpp_ga_instance(crossover: Callable, payload, context) -> float:
    idx, instance = payload
    np.random.seed(context["seed"] + idx)
    return _run_dpp_ga_instance(crossover, instance, context)


class DPPGAEvaluation(Evaluation):
    """Evaluator for ReEvo's DPP genetic-algorithm crossover task."""

    def __init__(
            self,
            timeout_seconds=300,
            split: str = DEFAULT_SPLIT,
            n_pop: int | None = None,
            n_iter: int | None = None,
            n_decap: int | None = None,
            elite_rate: float | None = None,
            seed: int = 5678,
            max_instances: int | None = None,
            eval_workers: int = 1,
            eval_backend: str = "sequential",
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        instances, metadata = load_split_instances(split=split)
        if max_instances is not None:
            instances = instances[:max_instances]
            metadata = {**metadata, "n_instances": len(instances)}
        self._instances = instances
        self.dataset_metadata = metadata
        self.n_instance = int(metadata["n_instances"])

        params = metadata["parameters"]
        self.n = int(params["grid_shape"][0])
        self.m = int(params["grid_shape"][1])
        self.freq_pts = int(params["freq_pts"])
        self.n_pop = int(n_pop if n_pop is not None else params["n_pop"])
        self.n_iter = int(n_iter if n_iter is not None else metadata["n_iter"])
        self.n_decap = int(n_decap if n_decap is not None else params["n_decap"])
        self.elite_rate = float(elite_rate if elite_rate is not None else params["elite_rate"])
        self.seed = int(seed)
        self._arrays: dict[str, np.ndarray] | None = None
        self.eval_workers = max(1, int(eval_workers))
        self.eval_backend = validate_backend(eval_backend, daemon_eval_process=self.daemon_eval_process)
        if self.eval_backend == "thread":
            raise ValueError("DPPGAEvaluation supports sequential or process eval_backend only.")

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = load_problem_arrays()
        return self._arrays

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        if callable_func is not None and (
                self.eval_backend == "sequential"
                or self.eval_workers == 1
                or self.n_instance <= 1
        ):
            return self.evaluate(callable_func)

        try:
            rewards = evaluate_instances(
                program_str=program_str,
                callable_func=callable_func,
                payloads=list(enumerate(self._instances)),
                instance_eval=_evaluate_dpp_ga_instance,
                context={
                    "arrays": self.arrays,
                    "n": self.n,
                    "m": self.m,
                    "freq_pts": self.freq_pts,
                    "n_pop": self.n_pop,
                    "n_iter": self.n_iter,
                    "n_decap": self.n_decap,
                    "elite_rate": self.elite_rate,
                    "seed": self.seed,
                },
                backend=self.eval_backend,
                workers=self.eval_workers,
                timeout_seconds=self.timeout_seconds,
            )
            return float(np.mean(rewards))
        except Exception:
            return None

    def evaluate(self, crossover: Callable) -> float | None:
        try:
            np.random.seed(self.seed)
            rewards = [
                self._run_instance(crossover, instance)
                for instance in self._instances
            ]
            return float(np.mean(rewards))
        except Exception:
            return None

    def _run_instance(self, crossover: Callable, instance: dict[str, Any]) -> float:
        return _run_dpp_ga_instance(
            crossover,
            instance,
            {
                "arrays": self.arrays,
                "n": self.n,
                "m": self.m,
                "freq_pts": self.freq_pts,
                "n_pop": self.n_pop,
                "n_iter": self.n_iter,
                "n_decap": self.n_decap,
                "elite_rate": self.elite_rate,
            },
        )

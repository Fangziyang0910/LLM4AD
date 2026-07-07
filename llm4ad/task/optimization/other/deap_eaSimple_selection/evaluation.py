from __future__ import annotations

import random
from typing import Any, Callable

import numpy as np
from deap import algorithms as deap_algorithms
from deap import base, creator, tools

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.other.deap_eaSimple_selection.dataset import load_split_instances
from llm4ad.task.optimization.other.deap_eaSimple_selection.template import task_description, template_program

__all__ = ["EASimpleSelectionEvaluation"]

if not hasattr(creator, "FitnessMinEASimpleSelection"):
    creator.create("FitnessMinEASimpleSelection", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "IndividualEASimpleSelection"):
    creator.create("IndividualEASimpleSelection", list, fitness=creator.FitnessMinEASimpleSelection)


class EASimpleSelectionEvaluation(Evaluation):
    """Evaluator for EoH's DEAP eaSimple parent-selection design task."""

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
        self.tournament_size = int(self.dataset_metadata["tournament_size"])
        self.cxpb = float(self.dataset_metadata["cxpb"])
        self.mutpb = float(self.dataset_metadata["mutpb"])
        self.eta_c = float(self.dataset_metadata["eta_c"])
        self.eta_m = float(self.dataset_metadata["eta_m"])

    def _run_ea_simple(
            self,
            instance: dict[str, Any],
            select_fn: Callable[[np.ndarray, int, int], np.ndarray],
            seed: int,
    ) -> float:
        random.seed(seed)
        np.random.seed(seed)

        func = instance["func"]
        dim = int(instance["dim"])
        lo, hi = instance["bounds"]
        indpb = 1.0 / dim

        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.uniform, lo, hi)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.IndividualEASimpleSelection,
            toolbox.attr_float,
            n=dim,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", lambda ind: (func(np.array(ind)),))
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=lo, up=hi, eta=self.eta_c)
        toolbox.register("mutate", tools.mutPolynomialBounded, low=lo, up=hi, eta=self.eta_m, indpb=indpb)

        def deap_select(individuals, k):
            fitnesses = np.array([ind.fitness.values[0] for ind in individuals])
            indices = np.asarray(select_fn(fitnesses, k, self.tournament_size), dtype=int)
            if indices.shape != (k,):
                raise ValueError(f"select returned shape {indices.shape}, expected ({k},)")
            if not np.all((indices >= 0) & (indices < len(individuals))):
                raise ValueError("select returned out-of-range indices")
            return [individuals[int(i)] for i in indices]

        toolbox.register("select", deap_select)

        pop = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        deap_algorithms.eaSimple(
            pop,
            toolbox,
            cxpb=self.cxpb,
            mutpb=self.mutpb,
            ngen=self.n_gen,
            halloffame=hof,
            verbose=False,
        )
        return float(hof[0].fitness.values[0])

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, select_fn: Callable[[np.ndarray, int, int], np.ndarray]) -> float | None:
        try:
            scores = []
            for instance in self._instances:
                run_bests = [
                    self._run_ea_simple(instance, select_fn, seed)
                    for seed in range(self.n_runs)
                ]
                scores.append(float(np.log1p(np.mean(run_bests))))
            return -float(np.mean(scores))
        except Exception:
            return None

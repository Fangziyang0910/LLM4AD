template_program = '''
import numpy as np


def select(fitnesses: np.ndarray, k: int, tournament_size: int) -> np.ndarray:
    """Select parent indices for DEAP's eaSimple GA.

    Args:
        fitnesses: Objective values for each individual. Lower is better.
        k: Number of parents to select.
        tournament_size: Reference tournament size supplied by the evaluator.

    Returns:
        Integer array of selected parent indices with shape (k,).
    """
    pop_size = len(fitnesses)
    selected = np.empty(k, dtype=int)
    for i in range(k):
        candidates = np.random.choice(pop_size, tournament_size, replace=False)
        selected[i] = candidates[np.argmin(fitnesses[candidates])]
    return selected
'''

task_description = (
    "Design a parent selection operator for DEAP's eaSimple genetic algorithm "
    "on minimization benchmarks. The evaluator fixes SBX crossover, polynomial "
    "mutation, and full generational replacement; only the selection operator is "
    "evolved. The function receives the current fitness array, the number of "
    "parents to select, and a reference tournament size, then returns selected "
    "parent indices. Performance is measured by negative mean log1p final best "
    "objective across continuous benchmark functions."
)

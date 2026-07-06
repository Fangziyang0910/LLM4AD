template_program = '''
import numpy as np


def mutation(population: np.ndarray, current_idx: int, best_idx: int,
             fitness: np.ndarray, F: float, bounds: np.ndarray) -> np.ndarray:
    """Return a mutant vector for Differential Evolution.

    Args:
        population: current population with shape (pop_size, dim).
        current_idx: index of the target individual being evolved.
        best_idx: index of the current best individual, lower fitness is better.
        fitness: objective value for each individual.
        F: mutation scale factor.
        bounds: array with shape (dim, 2), [lower, upper] per variable.

    Returns:
        A mutant vector with shape (dim,). The evaluator applies binomial
        crossover, bound clipping, and greedy selection.
    """
    pop_size, dim = population.shape
    candidates = [i for i in range(pop_size) if i != current_idx]
    r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
    return population[r1] + F * (population[r2] - population[r3])
'''

task_description = (
    "Design a mutation operator for Differential Evolution. The function receives "
    "the current population, target index, best index, fitness array, mutation scale "
    "factor, and variable bounds, and must return a mutant vector. The evaluator then "
    "uses standard binomial crossover and greedy selection. The baseline is DE/rand/1. "
    "Performance is measured by final objective value on Sphere, Rastrigin, Ackley, "
    "Rosenbrock, and Griewank minimization benchmarks."
)

template_program = '''
import numpy as np


def solve_metaheuristic(func, dim: int, bounds: np.ndarray, budget: int) -> np.ndarray:
    """Run a black-box search and return the best solution found.

    Args:
        func: objective callable f(x: np.ndarray) -> float, lower is better.
        dim: number of decision variables.
        bounds: array with lower and upper bounds, shape (2, dim).
        budget: maximum number of objective evaluations.

    Returns:
        Best solution vector with shape (dim,).
    """
    lower, upper = bounds[0], bounds[1]
    best_x = lower + (upper - lower) * np.random.rand(dim)
    best_f = func(best_x)
    for _ in range(max(0, budget - 1)):
        x = lower + (upper - lower) * np.random.rand(dim)
        fx = func(x)
        if fx < best_f:
            best_x = x
            best_f = fx
    return best_x
'''

task_description = (
    "Design a complete single-objective black-box metaheuristic for continuous "
    "minimization benchmarks. The program receives an objective function, dimension, "
    "box bounds, and evaluation budget, then returns the best solution found. The "
    "source EoH task used a Metaheuristic class; this LLM4AD task exposes the same "
    "solve contract as a single function for compatibility with the platform sampler."
)

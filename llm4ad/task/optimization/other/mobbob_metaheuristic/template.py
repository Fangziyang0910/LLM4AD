template_program = '''
import numpy as np


def solve_multiobjective(func, dim: int, bounds: np.ndarray,
                         budget: int, n_obj: int) -> np.ndarray:
    """Return a Pareto-front approximation for a black-box multi-objective problem.

    Args:
        func: objective callable f(x) -> objective vector, lower is better.
        dim: number of decision variables.
        bounds: array with lower and upper bounds, shape (2, dim).
        budget: maximum number of objective evaluations.
        n_obj: number of objectives.

    Returns:
        Decision vectors with shape (k, dim), where k >= 1.
    """
    lower, upper = bounds[0], bounds[1]
    archive_x, archive_f = [], []
    for _ in range(budget):
        x = lower + (upper - lower) * np.random.rand(dim)
        f = func(x)
        archive_x.append(x.copy())
        archive_f.append(f.copy())

    X = np.array(archive_x)
    F = np.array(archive_f)
    keep = np.ones(len(F), dtype=bool)
    for i in range(len(F)):
        dominated_by = np.all(F <= F[i], axis=1) & np.any(F < F[i], axis=1)
        dominated_by[i] = False
        if dominated_by.any():
            keep[i] = False
    return X[keep]
'''

task_description = (
    "Design a multi-objective black-box metaheuristic. The program receives an "
    "objective function, dimension, bounds, evaluation budget, and objective count, "
    "then returns a set of decision vectors approximating the Pareto front. The source "
    "EoH task used a Metaheuristic class; this LLM4AD task exposes the same solve "
    "contract as a single function for compatibility with the platform sampler. "
    "Performance is measured by 2D hypervolume on fixed ZDT benchmark configurations."
)

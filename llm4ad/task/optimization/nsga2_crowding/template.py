template_program = '''
import numpy as np


def crowding_distance(F: np.ndarray) -> np.ndarray:
    """Compute a diversity score for one non-dominated front.

    Args:
        F: objective vectors of one front, shape (n_solutions, n_objectives).

    Returns:
        Diversity score per solution, shape (n_solutions,). Higher is preferred.
    """
    n, m = F.shape
    distance = np.zeros(n)
    for obj in range(m):
        order = np.argsort(F[:, obj])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = F[order[-1], obj] - F[order[0], obj]
        if span < 1e-10:
            continue
        for k in range(1, n - 1):
            distance[order[k]] += (F[order[k + 1], obj] - F[order[k - 1], obj]) / span
    return distance
'''

task_description = (
    "Design a diversity metric for NSGA-II. The evaluator runs a complete NSGA-II "
    "loop on ZDT benchmark problems and calls the function for each non-dominated "
    "front. Higher diversity scores are preferred when selecting survivors and "
    "parents after Pareto rank. Performance is measured by final hypervolume on "
    "fixed train/test configurations."
)

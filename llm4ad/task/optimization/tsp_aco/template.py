template_program = '''
import numpy as np


def heuristics(distance_matrix: np.ndarray) -> np.ndarray:
    """Return edge desirability values for TSP ant colony optimization.

    Args:
        distance_matrix: n by n matrix of pairwise distances.

    Returns:
        A non-negative n by n matrix. Larger values make an edge more likely
        to be selected by ants during tour construction.
    """
    return 1.0 / (distance_matrix + 1e-9)
'''

task_description = (
    "Given a TSP distance matrix, design a heuristic matrix for ant colony "
    "optimization. The heuristic indicates how promising each edge is for tour "
    "construction. The goal is to minimize the best tour length found by ACO."
)

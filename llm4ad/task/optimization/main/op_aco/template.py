template_program = '''
import numpy as np


def heuristics(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    """Return edge desirability values for OP ant colony optimization.

    Args:
        prize: node prize vector. Node 0 is the depot.
        distance: n by n distance matrix.
        maxlen: maximum tour length budget.

    Returns:
        A non-negative n by n matrix. Larger values make an edge more likely
        to be selected by ants during route construction.
    """
    return prize[np.newaxis, :] / (distance + 1e-9)
'''

task_description = (
    "Given node prizes, pairwise distances, and a maximum route length for the "
    "Orienteering Problem, design a heuristic matrix for ant colony optimization. "
    "The depot is node 0. The goal is to maximize the collected prize while "
    "respecting the route length budget."
)

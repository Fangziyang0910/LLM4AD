template_program = '''
import numpy as np


def heuristics(
        distance_matrix: np.ndarray,
        coordinates: np.ndarray,
        demands: np.ndarray,
        capacity: int,
) -> np.ndarray:
    """Return edge desirability values for CVRP ant colony optimization.

    Args:
        distance_matrix: n by n matrix of pairwise distances.
        coordinates: n by 2 matrix of node coordinates. Node 0 is the depot.
        demands: vector of node demands. Demand of the depot is 0.
        capacity: vehicle capacity.

    Returns:
        A non-negative n by n matrix. Larger values make an edge more likely
        to be selected by ants during route construction.
    """
    return 1.0 / (distance_matrix + 1e-9)
'''

task_description = (
    "Given a CVRP distance matrix, node coordinates, customer demands, and "
    "vehicle capacity, design a heuristic matrix for ant colony optimization. "
    "The depot is node 0. The goal is to minimize the best route length found "
    "by ACO."
)

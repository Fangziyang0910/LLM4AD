template_program = '''
import numpy as np


def update_edge_distance(edge_distance: np.ndarray,
                         local_opt_tour: np.ndarray,
                         edge_n_used: np.ndarray) -> np.ndarray:
    """Modify the edge distance matrix to escape the current local optimum.

    Args:
        edge_distance: Symmetric matrix of original pairwise distances.
        local_opt_tour: Array of node indices forming the current local-optimal tour.
        edge_n_used: Symmetric matrix counting how many times each edge has been penalized.

    Returns:
        Updated distance matrix to guide the next local search.
    """
    return edge_distance.copy()
'''

task_description = (
    "Given a local-optimal TSP tour and the original edge distance matrix, design "
    "a strategy to update the distance matrix so that guided local search escapes "
    "the current local optimum. The modified distances steer local search toward "
    "unexplored regions of the solution space. Performance is measured by negative "
    "optimality gap against known optimal tour costs."
)

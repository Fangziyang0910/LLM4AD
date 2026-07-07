template_program = '''
import numpy as np


def destroy_nodes(current_tour: np.ndarray,
                  distance_matrix: np.ndarray,
                  n_destroy: int) -> np.ndarray:
    """Select nodes to remove from the current tour in the ruin phase.

    Args:
        current_tour: Open tour as node indices, without the closing return to start.
        distance_matrix: Symmetric pairwise distance matrix.
        n_destroy: Number of nodes to remove.

    Returns:
        Node indices to remove, with length at least n_destroy.
    """
    return np.random.choice(current_tour, size=n_destroy, replace=False)
'''

task_description = (
    "Design a destroy operator for a TSP ruin-and-recreate algorithm. At each "
    "iteration the evaluator removes selected nodes from the current best tour, "
    "reinserts them by cheapest insertion, then applies 2-opt local search. The "
    "goal is to minimize the final tour length on fixed Euclidean TSP instances."
)

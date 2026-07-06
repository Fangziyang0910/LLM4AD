template_program = '''
import numpy as np


def heuristics(demand: np.ndarray, capacity: int) -> np.ndarray:
    """Return item-pair desirability values for offline BPP ACO.

    Args:
        demand: item sizes with shape (n,).
        capacity: capacity of every bin.

    Returns:
        An n by n matrix where entry (i, j) indicates how promising it is to
        put item i and item j in the same bin.
    """
    return np.tile(demand / demand.max(), (demand.shape[0], 1))
'''

task_description = (
    "Given item sizes and a bin capacity for an offline bin packing problem, "
    "design a heuristic matrix for ant colony optimization. Each entry "
    "indicates how promising it is to place two items in the same bin. The "
    "goal is to minimize the number of bins used."
)

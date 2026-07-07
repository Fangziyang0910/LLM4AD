template_program = '''
import numpy as np


def score(item: int, bins: np.ndarray) -> np.ndarray:
    """Score feasible bins for assigning the current item.

    Args:
        item: Size of the current item.
        bins: Remaining capacities of feasible bins, all at least item.

    Returns:
        Priority score for each feasible bin. Higher is preferred.
    """
    return bins
'''

task_description = (
    "Design a scoring heuristic for online bin packing. Items arrive one at a "
    "time; for each item, the evaluator calls the score function on feasible bins "
    "and assigns the item to the bin with maximum score. The objective is to "
    "minimize excess bin usage over the L1 lower bound on fixed Weibull datasets."
)

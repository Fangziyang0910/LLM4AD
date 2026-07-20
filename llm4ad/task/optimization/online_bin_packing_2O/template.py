template_program = '''
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    return item - bins
'''

task_description = (
    "Implement a priority function for bi-objective online one-dimensional bin packing. "
    "Items arrive one by one and must be placed immediately into a bin with enough remaining capacity. "
    "The function receives the item size and an array of remaining capacities of currently feasible bins, "
    "and must return a priority score for each feasible bin. The item is placed into the bin with the "
    "highest priority. Evaluation returns a two-objective score that jointly minimizes the number of "
    "bins used and the packing runtime."
)

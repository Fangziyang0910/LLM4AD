import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    bins_float = bins.astype(float)
    return np.where(bins_float < item * 2.0, -(bins_float - item) ** 2, -bins_float)

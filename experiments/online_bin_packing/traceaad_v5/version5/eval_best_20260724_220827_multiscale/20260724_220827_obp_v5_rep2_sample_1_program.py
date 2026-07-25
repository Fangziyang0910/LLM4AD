import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # For best fit: prefer bins with smallest remaining capacity after placing item
    # residual = bins - item
    # priority = -residual (so smaller residual -> higher priority)
    residual = bins - item
    # Only bins where residual >= 0 are feasible; for others, we can set a very low priority
    # But per contract, bins array contains only feasible bins (remaining capacities that are >= item)
    # So all bins in the input are feasible.
    return -residual

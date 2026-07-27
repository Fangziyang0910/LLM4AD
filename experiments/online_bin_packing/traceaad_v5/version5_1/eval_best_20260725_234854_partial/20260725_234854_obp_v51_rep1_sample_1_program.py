import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Remaining space after placing item in each bin
    remaining_after = bins.astype(np.float64) - item
    
    # Best Fit: prefer bins with least remaining space (smallest waste)
    # So we want to minimize remaining_after, which means higher priority for smaller remaining_after
    # We use negative remaining_after so that smaller remaining_after gives higher priority
    priority = -remaining_after
    
    return priority

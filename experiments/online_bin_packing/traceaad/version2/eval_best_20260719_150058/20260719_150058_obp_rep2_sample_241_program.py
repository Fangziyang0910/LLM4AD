import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Calculate remaining capacity after inserting the item
    remaining = bins - item
    
    # Threshold for tight fit
    threshold = item * 0.65
    
    # Compute priorities using nested np.where
    # -np.inf where remaining < 0
    # -np.exp(remaining ** 2 * 0.3) where remaining < threshold (and >= 0)
    # -remaining otherwise (where remaining >= threshold)
    priorities = np.where(
        remaining < 0,
        -np.inf,
        np.where(
            remaining < threshold,
            -np.exp(remaining ** 2 * 0.3),
            -remaining
        )
    )
    
    return priorities

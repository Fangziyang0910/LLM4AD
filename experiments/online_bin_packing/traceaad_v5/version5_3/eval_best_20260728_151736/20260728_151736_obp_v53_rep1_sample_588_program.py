import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Calculate waste if item is placed in each bin
    waste = bins.astype(float) - item
    
    # Return -waste for tight fits (waste < 0.1 * bins), otherwise return -1e30
    result = np.where(waste < 0.1 * bins.astype(float), -waste, -1e30)
        
    return result

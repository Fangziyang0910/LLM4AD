import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Only bins with capacity >= item are feasible.
    # We compute the residual space after placing the item: residual = bin_capacity - item
    # Best fit: prefer bins where residual is minimized (i.e., tightest fit).
    # So priority = -residual (higher priority for tighter fit)
    # Add a tiny random component to break ties and avoid pathological cases.
    
    # Compute residual as float
    residual = np.asarray(bins, dtype=np.float64) - item
    
    # Priority: negative residual (so smaller residual = higher priority)
    # Add small random noise for tie-breaking
    noise = np.random.random(bins.shape) * 1e-9
    priority_scores = -residual + noise
    
    return priority_scores

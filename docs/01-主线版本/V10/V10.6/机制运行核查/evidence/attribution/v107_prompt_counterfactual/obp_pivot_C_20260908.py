import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    b = np.asarray(bins, dtype=float)
    item_f = float(item)
    
    feasible = b >= item_f
    
    # Initialize with a very low finite value for infeasible bins
    scores = np.full(b.shape, -1e12, dtype=float)
    
    if not np.any(feasible):
        return scores
    
    # Calculate slack: remaining space after placing the item
    slack = b[feasible] - item_f
    
    # Primary: favor larger slack (Worst-Fit)
    # We want to maximize slack, so score increases with slack
    primary = slack
    
    # Adaptive penalty: penalize placing small items in bins that leave a lot of space?
    # Actually, standard Worst-Fit just maximizes slack.
    # Let's add a small secondary preference for larger bins (capacity) to break ties
    secondary = 0.001 * b[feasible]
    
    # Combine
    local_scores = primary + secondary
    
    scores[feasible] = local_scores
    
    return scores
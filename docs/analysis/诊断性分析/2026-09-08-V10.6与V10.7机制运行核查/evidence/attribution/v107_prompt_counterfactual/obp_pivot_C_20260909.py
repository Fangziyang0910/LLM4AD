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

    # Base scores for infeasible bins
    scores = np.full(b.shape, -1e9, dtype=float)

    if np.any(feasible):
        slack = b[feasible] - item_f
        
        # Add small epsilon to avoid division by zero, favoring exact fits
        # If slack is 0, this gives a very high (negative but closest to 0? No, we want highest priority)
        # We want higher score for smaller slack.
        # Score = -1/(slack + eps). 
        # If slack=0, score = -1/eps (large negative).
        # If slack=1, score = -1/(1+eps).
        # This is confusing. Let's use: Score = - (1 / (slack + 1)).
        # slack=0 -> -1
        # slack=1 -> -0.5
        # slack=10 -> -0.09
        # This prefers smaller slack.
        
        # To make it stronger, we can use a power or scale.
        # Let's try: Score = -100 / (slack + 1)
        
        epsilon = 1e-6
        local_scores = -100.0 / (slack + epsilon)
        
        # Add a small linear component to break ties in favor of larger bins?
        # Or keep it pure non-linear Best-Fit.
        
        scores[feasible] = local_scores

    return scores
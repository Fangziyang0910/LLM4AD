import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Cast to float to allow safe arithmetic and ensure finite output type
    b = np.asarray(bins, dtype=float)
    item_f = float(item)
    
    # Identify feasible bins: those with enough remaining capacity
    feasible = b >= item_f
    
    # Initialize all scores to a very low finite value
    # This ensures infeasible bins are never selected if any feasible bin exists
    scores = np.full(b.shape, -1e12, dtype=float)
    
    if np.any(feasible):
        # Calculate slack: the remaining space left after placing the item
        # Slack is minimized for the "best fit" bin
        slack = b[feasible] - item_f
        
        # Primary score: Negative slack
        # Maximizing this score minimizes slack, which is the Best-Fit strategy
        # This effectively picks the bin with the smallest capacity that is >= item
        primary_score = -slack
        
        # Optional: Very tiny deterministic tie-breaker to handle exact ties
        # Since slack = b - item, equal slack implies equal b.
        # In case of floating point precision issues or identical integer bins,
        # we add a minuscule index-based penalty to ensure deterministic selection
        # that doesn't affect the primary ordering.
        # We prefer earlier indices (lower index = higher priority) for stability.
        indices = np.where(feasible)[0]
        tie_breaker = -1e-9 * indices
        
        scores[feasible] = primary_score + tie_breaker
        
    return scores
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Cast to float to ensure floating-point arithmetic and avoid integer issues
    b = np.asarray(bins, dtype=np.float64)
    item_f = float(item)
    
    # Identify feasible bins (remaining capacity >= item size)
    feasible = b >= item_f
    
    # Initialize scores with a very low finite value for infeasible bins
    # This ensures they are never selected if any feasible bin exists
    scores = np.full(b.shape, -1e12, dtype=np.float64)
    
    if np.any(feasible):
        # Calculate slack: remaining space after placing the item
        slack = b[feasible] - item_f
        
        # Primary term: Best-Fit logic.
        # Minimize slack by assigning -slack. Higher score (less negative) means tighter fit.
        primary_score = -slack
        
        # Secondary term: Tiny index-based tie-breaker.
        # This ensures deterministic behavior and breaks ties by preferring earlier indices
        # (or simply providing a unique finite value) without affecting the primary Best-Fit order
        # since the coefficient is extremely small.
        idx = np.arange(np.sum(feasible), dtype=np.float64)
        tie_breaker = 1e-12 * idx
        
        # Combine scores
        local_scores = primary_score + tie_breaker
        
        scores[feasible] = local_scores

    return scores
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Cast to float to avoid integer overflow/underflow and enable fractional scoring
    b = np.asarray(bins, dtype=float)
    item_f = float(item)
    
    # Identify feasible bins (remaining capacity >= item size)
    feasible = b >= item_f
    
    # Initialize scores with a very low finite value for infeasible bins
    scores = np.full(b.shape, -1e12, dtype=float)
    
    if np.any(feasible):
        # Calculate slack: remaining space after placing the item
        slack = b[feasible] - item_f
        
        # Primary term: Tightness reward. Use reciprocal to strongly favor minimal slack.
        # 1/(1+slack) is high when slack is small, low when slack is large.
        tightness = 1.0 / (1.0 + slack)
        
        # Secondary term: Capacity bonus. Prefer larger bins to preserve flexibility
        # for future larger items. Scale relative to max capacity to keep it bounded.
        max_cap = np.max(b[feasible]) if len(b[feasible]) > 0 else 1.0
        if max_cap > 0:
            capacity_bonus = 0.1 * (b[feasible] / max_cap)
        else:
            capacity_bonus = np.zeros_like(slack)
        
        # Tertiary term: Tiny tie-breaker to ensure deterministic ordering
        tie_breaker = 1e-9 * b[feasible]
        
        # Combine scores: Higher is better
        local_scores = tightness + capacity_bonus + tie_breaker
        scores[feasible] = local_scores
    
    return scores
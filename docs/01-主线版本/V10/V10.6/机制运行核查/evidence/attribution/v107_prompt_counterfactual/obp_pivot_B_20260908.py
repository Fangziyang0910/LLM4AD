import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Cast to float to ensure correct arithmetic and finite output type
    b = np.asarray(bins, dtype=float)
    item_f = float(item)
    
    # Identify feasible bins: those with enough remaining capacity
    feasible = b >= item_f
    
    # Initialize with a very low finite penalty for infeasible bins
    # This ensures they are never selected if any feasible bin exists
    scores = np.full(b.shape, -1e9, dtype=float)
    
    if np.any(feasible):
        # Slack is the space left over after placing the item
        # Minimizing slack is the Best-Fit strategy
        slack = b[feasible] - item_f
        
        # Primary term: -slack (higher is better, i.e., smaller slack)
        # Secondary term: tiny bonus for larger capacity to break ties
        # Preferring larger bins on ties can be slightly beneficial for balancing
        # overall bin utilization if capacities vary, though mathematically slack
        # uniqueness makes this rare in pure integer contexts. It adds stability.
        tie_breaker = 1e-6 * b[feasible]
        
        local_scores = -slack + tie_breaker
        scores[feasible] = local_scores
        
    # Ensure all values are finite (safety check, though logic above ensures this)
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
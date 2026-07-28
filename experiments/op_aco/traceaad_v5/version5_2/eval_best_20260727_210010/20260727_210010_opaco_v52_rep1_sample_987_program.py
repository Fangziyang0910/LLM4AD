import numpy as np
def heuristics(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    """Return edge desirability values for OP ant colony optimization.

    Args:
        prize: Node prizes with shape (n,). Node 0 is the depot.
        distance: Pairwise Euclidean distances with shape (n, n).
            Diagonal entries are large sentinels so self-loops are unused.
        maxlen: Maximum allowed tour length (return-to-depot constrained).

    Returns:
        An (n, n) edge-prior matrix. Larger values make an edge more likely
        to be sampled. Values at or below zero are treated as 1e-9.
    """
    n = distance.shape[0]
    
    # Avoid division by zero if n is 0, though n>=1 is expected for depot + nodes
    if n == 0:
        return np.empty((0, 0))
        
    # Calculate the decay scale factor based on budget and problem size
    # Using maxlen / n as a characteristic length scale for the problem
    scale = maxlen / n
    # Prevent division by zero in exp argument if scale is 0
    if scale == 0:
        return np.zeros_like(distance)
        
    # Create source and destination prize broadcasts
    prize_source = prize[:, np.newaxis] # Shape (n, 1)
    prize_dest = prize[np.newaxis, :]   # Shape (1, n)
    
    # Apply cubic weighting to destination prize for primary signal amplification
    prize_dest_cubic = prize_dest ** 3.0
    
    # Interaction term: linear product of source and destination prizes
    # This amplifies edges between two high-value nodes, encouraging chaining of rewards
    interaction = prize_source * prize_dest
    
    # Asymmetric source-departure penalty: penalize leaving high-prize nodes
    # Modified: Use maxlen * 2.0 to soften the penalty gradient, allowing ants to
    # occasionally leave high-prize nodes if no better alternatives exist.
    source_penalty = np.exp(-prize_source / (maxlen * 2.0))
    
    with np.errstate(over='ignore', under='ignore'):
        rational_term = 1.0 / np.sqrt(distance**2 + 1.0)
        exponent = -distance / scale
        decay_factor = np.exp(exponent)
        
    # Combine: Interaction * Cubic Dest Weight * Source Penalty * Distance Penalties
    heuristic = interaction * prize_dest_cubic * source_penalty * rational_term * decay_factor
    
    # Ensure values are finite and positive as per contract
    # Replace any NaNs or Infs with a small safe value
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    # Ensure strict positivity for ACO sampling (values <= 0 treated as 1e-9)
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic

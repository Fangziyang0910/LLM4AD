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
    # Compute the return cost for each node to keep shape compatible (n, 1)
    return_cost = distance[:, 0:1]
    
    # Compute denominator using addition for smoother gradient transitions
    denom = distance + 1e-10
    
    # Rank-based prize normalization for base exponent
    # Using double argsort to handle ties (duplicate prizes) robustly
    n = len(prize)
    if n > 1:
        ranks = np.argsort(np.argsort(prize)).astype(float) / (n - 1)
    else:
        ranks = np.zeros(n)
        
    # Global mean-normalized prize ratio for stability
    # Use global mean instead of pairwise source prize to reduce sensitivity to outliers
    mean_prize = np.mean(prize[1:]) if n > 1 else prize[0]
    if mean_prize == 0:
        mean_prize = 1e-10
        
    # prize_ratio_normalized[j] = prize[j] / mean_prize
    # Broadcast to (n, n) matrix where entry [i, j] is prize[j] / mean_prize
    # This provides a global context rather than local pairwise comparison
    prize_ratio_normalized = prize[np.newaxis, :] / mean_prize
    
    # Modify base exponent with global prize density
    base_exponent = 2.0 * (1 + np.clip(ranks[np.newaxis, :], 0, 0.5) * np.clip(prize_ratio_normalized, 0, 1.0))
    
    # Budget-tightness component
    # residual = maxlen - distance[i, j] - distance[j, 0]
    # tightness = clip(residual / distance, 0, 2)
    tightness = np.clip((maxlen - distance - return_cost) / denom, 0, 2)
    
    # Total exponent is the sum of base and tightness components
    exponent = base_exponent + tightness
    
    # Heuristic value: prize[j] / distance[i,j]^exponent
    heuristic_values = prize[np.newaxis, :] / np.power(denom, exponent)
    
    # Feasibility mask: edge (i, j) is feasible if distance[i, j] + distance[j, 0] <= maxlen
    # Which is equivalent to residual >= 0
    residual = maxlen - distance - return_cost
    feasible_mask = residual >= 0
    
    # Revert to static 1e-9 floor for infeasible edges
    return np.where(feasible_mask, heuristic_values, 1e-9)

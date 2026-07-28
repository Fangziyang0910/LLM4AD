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
    n = len(prize)
    
    # Create a matrix of prizes for all destination nodes (columns)
    # prize_matrix[i, j] = prize[j]
    prize_matrix = prize[np.newaxis, :]  # Shape (1, n)
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-10
    
    # Compute edge distances with epsilon added to avoid division issues
    dist_safe = distance + epsilon
    
    # Base efficiency: prize[j] * maxlen / distance[i, j]^2
    # This rewards high prizes and penalizes long distances quadratically
    eff = prize_matrix * maxlen / (dist_safe ** 2)
    
    # Create a copy for median calculation, excluding diagonal only
    # Inclusive global median normalization as per request
    eff_med = eff.copy()
    
    # Zero out diagonal
    np.fill_diagonal(eff_med, 0.0)
    
    # Compute global median of non-zero efficiencies from all non-diagonal edges
    flat_eff = eff_med.flatten()
    non_zero_eff = flat_eff[flat_eff > 0]
    
    if len(non_zero_eff) > 0:
        med = np.nanmedian(non_zero_eff)
    else:
        med = epsilon
        
    # Avoid division by zero in median
    med_safe = max(med, epsilon)
    
    # Normalize efficiency by median
    normalized = eff / med_safe
    
    # Dynamic exponent schedule:
    # - For normalized >= 1.0 (above median), use exponent 2.5 (strong pressure)
    # - For normalized < 1.0 (below median), interpolate exponent from 1.2 to 2.5
    #   This prevents excessive suppression of mid-tier moves while keeping high-value focus.
    
    # Identify regions where normalized < 1
    below_median_mask = normalized < 1.0
    
    # For those regions, calculate interpolated exponent
    # Ensure normalized is non-negative for the calculation (though it should be)
    safe_normalized = np.clip(normalized, 0, 1)
    
    # Calculate exponent using np.where for vectorized operation
    # exponent = 2.5 if normalized >= 1.0 else 1.2 + 1.3 * normalized
    exponent = np.where(below_median_mask, 1.2 + 1.3 * safe_normalized, 2.5)
    
    # Compute the scale factor: 1 + |normalized - 1|^exponent
    # Using abs to ensure real results for normalized < 1, avoiding NaNs from negative bases with fractional exponents.
    base = np.abs(normalized - 1.0)
    
    # Calculate scale using the dynamic exponent
    scale = 1.0 + base ** exponent
    
    heuristic = eff * scale
    
    # Apply soft budget-feasibility penalty
    # Penalize longer edges exponentially based on their fraction of the total budget
    # Using linear penalty as per Reference Program Step 6 which yielded peak performance
    # Factor: exp(-0.1 * distance / maxlen)
    budget_penalty = np.exp(-0.1 * distance / maxlen)
    heuristic = heuristic * budget_penalty
    
    # Set diagonal to 0 (no self-loops)
    np.fill_diagonal(heuristic, 0.0)
    
    # Ensure the heuristic value for edges leaving the depot (node 0) is strictly zero
    heuristic[0, :] = 0.0
    
    # Ensure all values are finite and positive (replace non-positive with a small positive value)
    heuristic = np.where(heuristic <= 0, 1e-9, heuristic)
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic

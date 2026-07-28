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
    
    # Create a mask for valid edges (non-diagonal, finite distance)
    mask = ~np.eye(n, dtype=bool) & np.isfinite(distance)
    
    # Prepare prize vector for broadcasting: prize[j] depends on column j
    prizes_j = prize[np.newaxis, :]  # Shape (1, n)
    
    # Extract distances to depot (column 0)
    dist_to_depot = distance[:, 0]  # Shape (n,)
    
    # Broadcast for matrix calculation
    d_i_0 = dist_to_depot[:, np.newaxis]  # Shape (n, 1), distance from node i to depot
    
    # Numerical stability constant
    eps = 1e-9
    
    # Calculate base heuristic: prize_j / dist^3
    # Use np.maximum for numerical stability to avoid division by zero or very small numbers.
    # Invalid edges (masked) will still be zeroed out later, so their heuristic value
    # before masking doesn't strictly matter for feasibility, but this ensures smooth gradients.
    safe_dist_cube = np.maximum(distance, eps) ** 3
    
    # Calculate asymmetric forward slack for all edges
    # forward_slack[i,j] = maxlen - d(i,0) - d(i,j)
    # This represents the budget remaining after leaving depot, visiting i, then moving to j.
    # It prioritizes edges that leave more budget for future exploration from j.
    forward_slack = maxlen - d_i_0 - distance
    
    # Normalize slack for interpolation
    # We want to interpolate between factor=1.0 (tight budget) and factor=sqrt(prize/mean) (loose budget)
    # Transition around 0.5 * maxlen
    half_maxlen = 0.5 * maxlen
    
    # Create a smooth transition function: t = 0 when slack < 0, t = 1 when slack > maxlen
    # Using a sigmoid-like shape or simple normalized ramp clipped to [0,1]
    # Let's use a quadratic smooth step for better gradient properties than linear
    # normalized_slack in [0, 1] roughly
    # raw_slack is already calculated. Let's normalize it relative to half_maxlen.
    # If slack is small (< 0), factor should be 1.
    # If slack is large (> maxlen), factor should be sqrt(prize/mean).
    # Let's map slack to [0, 1] using a range centered around half_maxlen.
    # A simple logistic function or a polynomial smoothstep can work.
    # Let's use: t = clip( (slack / half_maxlen) ** 2, 0, 1 ) ? 
    # Or simpler: t = clip( slack / half_maxlen, 0, 1 )
    # Let's try a quadratic ramp for smoother transition:
    # t = clip( (slack / half_maxlen), 0, 1 )^2
    
    normalized_slack = forward_slack / (half_maxlen + eps)
    # Smooth step: 0 when slack <= 0, 1 when slack >= half_maxlen
    # Using polynomial smoothstep: t = 3x^2 - 2x^3 for x in [0,1]
    # First clip to [0,1]
    t_clip = np.clip(normalized_slack, 0.0, 1.0)
    # Smoothstep
    alpha = 3.0 * t_clip**2 - 2.0 * t_clip**3
    
    # Calculate prize density factor
    mean_prize = np.mean(prize)
    prize_density_factor = np.sqrt(prizes_j / (mean_prize + eps))
    
    # Interpolate between 1.0 and prize_density_factor
    # When alpha=0 (tight), factor = 1.0
    # When alpha=1 (loose), factor = prize_density_factor
    interpolated_factor = 1.0 + alpha * (prize_density_factor - 1.0)
    
    H = prizes_j / safe_dist_cube * interpolated_factor
    
    # Use robust raw squared asymmetric forward slack
    # Clip to 0 to ensure infeasible edges (negative slack) contribute exactly zero
    slack_factor = np.maximum(forward_slack, 0.0) ** 2
    
    H = H * slack_factor
    
    # Zero out invalid edges explicitly to ensure they are exactly 0
    H[~mask] = 0.0
    
    # Row-wise sum normalization to create a local probability distribution
    # Use np.where to robustly handle rows with no feasible moves (sum=0)
    row_sum = np.sum(H, axis=1, keepdims=True)
    H = np.where(row_sum > 0, H / row_sum, 0.0)
        
    return H

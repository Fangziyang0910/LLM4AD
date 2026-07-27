
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
    epsilon = 1e-9

    if n < 2 or maxlen <= 0:
        return np.full((n, n), epsilon)

    # Precompute distances from each node to the depot (node 0)
    dist_to_depot = distance[:, 0]  # Shape (n,)
    dist_0_i = dist_to_depot[:, np.newaxis]  # (n, 1) for broadcasting with distance[i, j]
    dist_j_0 = dist_to_depot[np.newaxis, :]  # (1, n) for broadcasting with distance[i, j]

    # Safe distance matrix to avoid division by zero
    dist_safe = np.maximum(distance, epsilon)

    # --- Component 1: Prize Efficiency ---
    # Reward high prizes relative to the distance from the current node.
    prize_col = prize[np.newaxis, :]  # (1, n)
    prize_efficiency = prize_col / dist_safe  # (n, n)

    # --- Component 2: Clarke-Wright Savings Factor ---
    # Savings(i, j) = dist(0, i) + dist(0, j) - dist(i, j)
    # High savings indicate that connecting i and j is more efficient than serving them separately.
    savings = dist_0_i + dist_j_0 - distance  # (n, n)
    positive_savings = np.maximum(savings, 0.0)
    savings_efficiency = positive_savings / dist_safe  # (n, n)

    # --- Component 3: Budget Slack Factor ---
    # Calculate the cost of the tour segment 0 -> i -> j -> 0.
    cost_triangle = dist_0_i + distance + dist_j_0  # (n, n)
    
    # Slack is the remaining budget if we were to close the tour immediately after j.
    slack = maxlen - cost_triangle
    
    # We create a slack factor that is 1.0 when slack is maximal (cost is 0) and decreases linearly.
    # To prevent zeroing out valid edges that just fit, we clip slack at epsilon.
    # However, we also want to penalize edges that use too much budget.
    # A common approach is to normalize slack. 
    # If slack < 0, the edge is strictly infeasible for a immediate return, which we mask later.
    slack_clipped = np.maximum(slack, 0.0)
    
    # Normalize slack by maxlen to get a factor in [0, 1] roughly.
    # Adding epsilon to denominator prevents division by zero.
    slack_factor = slack_clipped / (maxlen + epsilon)

    # --- Combine Components ---
    # Heuristic = Prize_Efficiency * Savings_Efficiency * Slack_Factor
    raw_heuristic = prize_efficiency * savings_efficiency * slack_factor

    # --- Amplify ---
    # Use a power function to sharpen the distribution. 
    # Alpha=2.5 worked well in Alg 3. Let's try 2.8 to slightly increase discrimination 
    # without causing excessive numerical instability.
    alpha = 2.8
    heuristic_matrix = np.power(raw_heuristic, alpha)

    # --- Feasibility Mask ---
    # An edge i -> j is only feasible if the path 0 -> i -> j -> 0 fits within maxlen.
    # This ensures that the ant can always return to the depot if it stops here.
    feasible_mask = cost_triangle <= maxlen
    
    # Apply mask: infeasible edges get minimal value
    heuristic_matrix = np.where(feasible_mask, heuristic_matrix, epsilon)
    
    # Handle numerical stability: NaNs, Infs
    heuristic_matrix = np.nan_to_num(heuristic_matrix, nan=epsilon, posinf=1e9, neginf=epsilon)
    
    # Ensure all values are at least epsilon to avoid zero-probability edges in ACO sampling
    heuristic_matrix = np.maximum(heuristic_matrix, epsilon)
    
    return heuristic_matrix

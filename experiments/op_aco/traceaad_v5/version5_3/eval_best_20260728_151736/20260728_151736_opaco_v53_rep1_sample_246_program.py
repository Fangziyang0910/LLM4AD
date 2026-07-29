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
    
    # Extract distances to depot (column 0)
    dist_to_depot = distance[:, 0]  # Shape (n,)
    
    # Pairwise distances
    dist = distance  # (n, n)
    
    # Prize at destination node j for edge i->j
    # prize[j] for each column j
    dest_prize = prize[np.newaxis, :]  # (1, n)
    
    # Avoid division by zero for base desirability calculation
    eps = 1e-9
    
    # Base desirability: prize * exp(-dist / median_dist)
    # Scale is derived from the median distance in the instance
    # Flatten distance matrix and compute median.
    median_dist = np.median(dist)
    if median_dist == 0:
        median_dist = eps
    scale = median_dist
    
    # Exponential decay base desirability
    base_val = dest_prize * np.exp(-dist / scale)  # (n, n)
    
    # Feasibility factor: Global exponential penalty based on total tour cost commitment
    # Use 'total_cost' formulation: dist[i,j] + dist[j,depot] + dist[i,depot]
    # Normalize against global 'maxlen' for stability.
    # cost_commitment[i, j] represents the cost of extending the tour from i to j and returning to depot,
    # plus the cost incurred to reach i from depot (to measure total tour load relative to budget).
    cost_commitment = dist + dist_to_depot[np.newaxis, :] + dist_to_depot[:, np.newaxis]  # Shape (n, n)
    
    # Global feasibility ratio
    # Using maxlen as denominator provides a stable global scale.
    feasibility_ratio = cost_commitment / maxlen
    
    # Local prize density bonus: prize[j] / dist(i, j)
    # Higher bonus for high prize and short distance.
    safe_dist = dist + eps
    local_density = dest_prize / safe_dist
    
    # Normalize local density by median prize value for consistent scaling across instances
    median_prize = np.median(prize)
    if median_prize == 0:
        median_prize = eps
    density_term = local_density / median_prize
    
    # Combine the feasibility ratio and density bonus
    # The exponent penalizes high cost ratios (exp(-ratio)) and rewards high local density (+density_term)
    decay_exponent = -feasibility_ratio + density_term
    
    decay = np.exp(decay_exponent)
    
    # Combine
    heuristic_matrix = base_val * decay
    
    # Zero out self-loops
    np.fill_diagonal(heuristic_matrix, 0.0)
    
    # Ensure values at or below zero are treated as eps (though our calculation yields positive values)
    heuristic_matrix = np.maximum(heuristic_matrix, eps)
    
    return heuristic_matrix

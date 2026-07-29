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
    eps = 1e-9
    
    # Extract distances to depot (column 0)
    dist_to_depot = distance[:, 0]  # Shape (n,)
    
    # Pairwise distances
    dist = distance  # (n, n)
    
    # Prize at destination node j for edge i->j
    # prize[j] for each column j
    dest_prize = prize[np.newaxis, :]  # (1, n)
    
    # Base desirability: prize * exp(-dist / median_dist)
    # Scale is derived from the median distance in the instance for smoother adaptive decay
    median_dist = np.median(dist)
    if median_dist == 0:
        median_dist = eps
    scale = median_dist
    
    base_val = dest_prize * np.exp(-dist / scale)  # (n, n)
    
    # Unified exponent structure:
    # Cost commitment: sum of edge cost and destination-to-depot cost (simplified)
    cost_commitment = dist + dist_to_depot[np.newaxis, :]
    cost_ratio = cost_commitment / maxlen
    
    # Local density term: unnormalized prize-to-distance ratio
    safe_dist = dist + eps
    density_term = dest_prize / safe_dist
    
    # Combine feasibility and local urgency in a unified exponent
    # High density can partially offset high cost ratio
    unified_factor = np.exp(-cost_ratio + density_term)
    
    # Combine: Base desirability * Unified Factor
    heuristic_matrix = base_val * unified_factor
    
    # Zero out self-loops
    np.fill_diagonal(heuristic_matrix, 0.0)
    
    # Ensure values at or below zero are treated as eps
    heuristic_matrix = np.maximum(heuristic_matrix, eps)
    
    return heuristic_matrix

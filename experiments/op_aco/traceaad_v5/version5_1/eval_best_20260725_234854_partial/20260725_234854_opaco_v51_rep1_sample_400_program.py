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
    n = prize.shape[0]
    
    # Ensure numerical stability for distance operations
    # Using a small epsilon to avoid division by zero
    eps = 1e-10
    safe_dist = np.maximum(distance, eps)
    
    # Component 1: Cubic inverse-distance bias (prize / distance^3)
    # Use broadcasting instead of np.tile for efficiency:
    # prize[np.newaxis, :] has shape (1, n), safe_dist ** 3 has shape (n, n)
    # Result has shape (n, n) where entry [i, j] = prize[j] / safe_dist[i, j]**3
    prize_dist_ratio = prize[np.newaxis, :] / (safe_dist ** 3)
    
    # Component 2: Destination-specific residual budget heuristic using broadcasting
    # We estimate the cost of taking edge (i, j) and then returning to depot 0.
    # Round-trip estimate: distance[i, j] + distance[j, 0]
    # distance[:, 0] is the vector of distances from any node to the depot.
    dist_to_depot = distance[:, 0]  # shape (n,)
    
    # Use broadcasting: safe_dist is (n, n), dist_to_depot is reshaped to (n, 1)
    # Result shape: (n, n) where round_trip_cost[i, j] = safe_dist[i, j] + dist_to_depot[j]
    round_trip_cost = safe_dist + dist_to_depot[:, np.newaxis]
    
    # Pure quadratic residual budget factor
    # (np.maximum(maxlen - round_trip_cost, 0.0) / maxlen) ** 2
    residual_budget = np.maximum(maxlen - round_trip_cost, 0.0)
    budget_factor = (residual_budget / maxlen) ** 2
    
    # Component 3: Prize-density correction factor
    # (prize[j] / mean(prize)) ** 0.5
    # Calculate mean prize, avoiding zero division if all prizes are zero
    mean_prize = np.mean(prize)
    if mean_prize == 0:
        # If all prizes are zero, the correction factor is 1 (neutral)
        prize_density_factor = np.ones((1, n))
    else:
        # prize is shape (n,), we need (1, n) for broadcasting against (n, n)
        prize_density_factor = (prize[np.newaxis, :] / mean_prize) ** 0.5
    
    # Combined heuristic
    heuristic_matrix = prize_dist_ratio * budget_factor * prize_density_factor
    
    return heuristic_matrix

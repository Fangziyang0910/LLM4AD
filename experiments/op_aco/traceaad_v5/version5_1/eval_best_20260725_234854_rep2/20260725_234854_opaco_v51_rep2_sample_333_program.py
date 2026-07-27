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
    
    # Avoid division by zero
    epsilon = 1e-10
    
    # Compute prize matrix for j: shape (1, n)
    prize_j = prize[np.newaxis, :]
    
    # Distance matrix: shape (n, n)
    dist_ij = distance
    
    # Distance from j to depot: shape (n,) -> broadcast to (n, n) as columns
    # distance[:, 0] gives dist(j, depot) for each j
    dist_j_to_depot = distance[:, 0]  # shape (n,)
    dist_j_to_depot_matrix = dist_j_to_depot[np.newaxis, :]  # shape (1, n)
    
    # Compute the feasibility factor using a "soft" slack gradient.
    # The cost of taking edge (i, j) and immediately returning to depot is dist(i, j) + dist(j, 0).
    # We define the slack as maxlen - (dist(i, j) + dist(j, 0)).
    # We normalize this slack by maxlen, clip to [0, 1], and apply a cube root transformation
    # to provide a robust gradient for balancing exploration and constraint satisfaction.
    
    round_trip_cost = dist_ij + dist_j_to_depot_matrix  # shape (n, n)
    slack = maxlen - round_trip_cost
    
    # Normalize slack and apply cube root for robust gradient (Slack Scaling Stability)
    normalized_slack = np.clip(slack / maxlen, 0, 1) ** (1/3)
    
    # Dynamic distance-dependent exponent for robustness and sensitivity
    # Micro-Tuning: Reduced decay scale divisor from 4.0 to 3.8 to sharpen short-edge prioritization
    alpha = 2.5 + 1.5 * np.exp(-dist_ij / (maxlen / 3.8))
    
    # Compute heuristic matrix with prize-density structure:
    # heuristic = (prize[j] / dist[i,j])^alpha * normalized_slack
    
    # Compute prize/distance ratio
    # Add epsilon to denominator to avoid division by zero
    ratio = prize_j / (dist_ij + epsilon)
    
    # Raise to power alpha
    # ratio shape: (1, n), dist_ij shape: (n, n) -> ratio broadcast to (n, n)
    # alpha shape: (n, n)
    prize_density_power = np.power(ratio, alpha)
    
    # Combine components with normalized slack
    heuristic_matrix = prize_density_power * normalized_slack
    
    # Ensure finite values
    heuristic_matrix = np.where(np.isfinite(heuristic_matrix), heuristic_matrix, 1e-9)
    
    # Clamp small/negative values to a small positive number as per contract
    # "Values at or below zero are treated as 1e-9"
    heuristic_matrix = np.maximum(heuristic_matrix, 1e-9)
    
    return heuristic_matrix

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
    dist_ij = distance  # Shape (n, n)
    
    # Prize at destination node j for edge i->j
    # prize[j] for each column j
    dest_prize = prize[np.newaxis, :]  # (1, n)
    
    # Distance from i to j
    dist = distance  # (n, n)
    
    # Avoid division by zero for base desirability calculation
    eps = 1e-9
    
    # Base desirability: prize / distance^3
    # Using cube of distance penalizes long jumps even more heavily than squared
    dist_cu = dist ** 3
    safe_dist_cu = dist_cu + eps
    
    base_val = dest_prize / safe_dist_cu  # (n, n)
    
    # Feasibility factor: Dynamic exponential penalty based on remaining budget from node i
    # Cost commitment = dist(i, j) + dist(j, depot)
    # Remaining budget from node i = maxlen - dist(i, depot)
    # This penalizes edges that consume a large portion of the specific feasible tour length from node i.
    cost_commitment = dist + dist_to_depot[np.newaxis, :]  # Shape (n, n)
    
    # Dynamic remaining budget: maxlen - dist(i, depot)
    # We use np.maximum to ensure the denominator is positive and avoid division by zero/negative.
    # dist_to_depot[:, np.newaxis] is (n, 1).
    remaining_budget = np.maximum(maxlen - dist_to_depot[:, np.newaxis], eps)
    
    # Local prize density bonus: prize[j] / dist(i, j)
    # Higher bonus for high prize and short distance.
    safe_dist = dist + eps
    local_density = dest_prize / safe_dist
    
    # Normalize local density by remaining budget for dynamic scaling relative to resource margin
    density_term = local_density / remaining_budget
    
    # Exponent for dynamic decay
    decay_exponent = -(cost_commitment / remaining_budget) + density_term
    
    decay = np.exp(decay_exponent)
    
    # Combine
    heuristic_matrix = base_val * decay
    
    # Hard infeasibility mask:
    # Set heuristic to eps if cost_commitment > remaining_budget.
    # This strictly enforces that an edge is only considered if the node j 
    # can theoretically reach the depot within the budget starting from i.
    infeasible_mask = cost_commitment > remaining_budget
    heuristic_matrix[infeasible_mask] = eps
    
    # Zero out self-loops
    np.fill_diagonal(heuristic_matrix, 0.0)
    
    # Ensure values at or below zero are treated as eps (though our calculation yields positive values)
    heuristic_matrix = np.maximum(heuristic_matrix, eps)
    
    return heuristic_matrix

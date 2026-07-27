
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
    epsilon = 1e-10
    
    # 1. Precompute distances to depot for return cost calculation
    dist_to_depot = distance[:, 0]
    
    # 2. Calculate the "Completion Cost" for each edge i -> j
    completion_cost = distance + dist_to_depot[np.newaxis, :]
    
    # 3. Calculate Slack (Remaining Budget)
    slack = maxlen - completion_cost
    
    # 4. Feasibility Mask
    feasible_mask = slack >= 0
    
    # 5. Heuristic Component 1: Prize Efficiency with Power Law and Detour Penalty
    safe_distance = np.maximum(distance, epsilon)
    base_efficiency = prize[np.newaxis, :] / safe_distance
    
    # Calculate average distance to depot for normalization of detour penalty
    avg_dist_to_depot = np.mean(dist_to_depot)
    safe_avg_dist = np.maximum(avg_dist_to_depot, epsilon)
    
    # Detour penalty using average distance for normalization
    alpha = 0.4
    detour_penalty = np.exp(-distance / (alpha * safe_avg_dist))
    
    # Combine efficiency and detour penalty with moderate exponent (2.5) for preference
    efficiency_component = (base_efficiency ** 2.5) * detour_penalty
    
    # 6. Heuristic Component 2: Slack Potential (Cubic Root)
    # Using cubic root for moderate penalty on low slack while maintaining sensitivity
    slack_normalized = np.maximum(slack, 0.0) / (maxlen + epsilon)
    slack_potential = np.power(slack_normalized, 1/3)
    
    # 7. Combine Components
    heuristic_matrix = efficiency_component * slack_potential
    
    # 8. Apply Feasibility Mask
    heuristic_matrix = np.where(feasible_mask, heuristic_matrix, 0.0)
    
    # 9. Numerical Stability
    heuristic_matrix = np.maximum(heuristic_matrix, 0.0)
    heuristic_matrix = np.nan_to_num(heuristic_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    return heuristic_matrix

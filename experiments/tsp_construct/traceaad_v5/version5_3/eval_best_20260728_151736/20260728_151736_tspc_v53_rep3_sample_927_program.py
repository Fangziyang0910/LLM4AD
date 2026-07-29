import numpy as np
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 0:
        return destination_node
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    unvisited_indices = unvisited_nodes
    n_unvisited = len(unvisited_indices)
    
    # Estimate total nodes from matrix shape
    n_total = distance_matrix.shape[0]
    
    # Calculate dynamic destination weight based on tour progress
    # visited_fraction increases as n_unvisited decreases
    # Avoid division by zero
    if n_total > 0:
        visited_fraction = 1 - (n_unvisited / n_total)
    else:
        visited_fraction = 0.0
        
    # Non-linear destination weight: quadratic acceleration near end of tour
    # This creates a sharper bias towards the destination when the tour is nearly complete
    dest_weight = 1 + (visited_fraction ** 2) * 2
    
    # Distance from current node to each candidate
    dist_from_current = distance_matrix[current_node, unvisited_indices]
    
    # Calculate connectivity metrics for each candidate
    # Submatrix of distances among unvisited nodes
    unvisited_dist_matrix = distance_matrix[np.ix_(unvisited_indices, unvisited_indices)]
    
    # Sum of distances from each candidate to all other unvisited nodes
    # Diagonal is 0, so sum(axis=1) is sum to others
    sum_dist_to_others = np.sum(unvisited_dist_matrix, axis=1)
    
    # Distance from each candidate to destination
    dist_to_dest = distance_matrix[unvisited_indices, destination_node]
    
    # Calculate denominator: reachability/connectivity score
    # Higher is better (lower final score).
    # Combines sum of distances to others (connectivity to remaining tour)
    # and weighted distance to destination (closure feasibility)
    # Introduce logarithmic scaling to the destination weight term for smoother penalty increase
    log_scale = np.log(n_unvisited + 1)
    
    # Modification: Revert to raw sum for connectivity term to use quadratic scaling with tour length
    connectivity_term = sum_dist_to_others
        
    denominator = connectivity_term + dist_to_dest * dest_weight * log_scale
    
    # Calculate numerator: effective distance cost
    epsilon = 1e-9
    
    # Bottleneck Avoidance Heuristic
    # Calculate the minimum distance to the destination among the remaining unvisited nodes
    # for each candidate. We penalize selections that leave behind nodes poorly connected 
    # to the destination.
    
    # Find global minimum and second minimum distances to destination among unvisited nodes
    sorted_dists = np.sort(dist_to_dest)
    min_dist_all = sorted_dists[0]
    second_min_dist_all = sorted_dists[1] if n_unvisited > 1 else min_dist_all
    
    # For each candidate i, determine the min distance to destination of the remaining set
    # If the candidate has the unique minimum distance, the remaining min is the second smallest.
    # Otherwise, the global minimum remains.
    min_dist_remaining = np.where(
        dist_to_dest == min_dist_all,
        second_min_dist_all,
        min_dist_all
    )
    
    # Normalize the bottleneck penalty using Harmonic Mean instead of ratio
    avg_dist_to_dest = np.mean(dist_to_dest)
    
    # Harmonic mean of min_dist_remaining and avg_dist_to_dest
    # H = 2 * a * b / (a + b)
    harmonic_mean = 2 * min_dist_remaining * avg_dist_to_dest / (min_dist_remaining + avg_dist_to_dest + epsilon)
    
    # Scale by sqrt(n_unvisited) to maintain consistent influence across tour stages
    bottleneck_penalty = harmonic_mean * np.sqrt(n_unvisited)
    
    # Exponential progress-weighted penalty: intensifies sharply only in the final stages
    # np.exp(visited_fraction - 1) is close to 0 when visited_fraction is small,
    # and grows rapidly as visited_fraction approaches 1.
    dynamic_penalty_weight = 1 + np.exp(visited_fraction - 1)
    
    # Replace Linear Local Density Pressure with Cluster Escape Potential
    # This non-linearly rewards nodes that are poorly connected to the remaining cluster
    safe_sum_dist = np.maximum(sum_dist_to_others, epsilon)
    cluster_escape_potential = np.log1p(1.0 / (safe_sum_dist + epsilon)) * (n_unvisited - 1)
    
    # Combine cluster escape potential and bottleneck penalty in the numerator
    numerator = dist_from_current * cluster_escape_potential * (1 + bottleneck_penalty * dynamic_penalty_weight)
    
    # Final score
    safe_denominator = np.maximum(denominator, epsilon)
    scores = numerator / safe_denominator
    
    # Select candidate with minimum score
    min_idx = np.argmin(scores)
    
    return int(unvisited_indices[min_idx])

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
    
    # If only one unvisited node, must go there
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    # Number of remaining nodes to visit
    num_remaining = len(unvisited_nodes)
    
    # Get distances from current node to all unvisited candidates
    current_distances = distance_matrix[current_node, unvisited_nodes]
    
    # Extract submatrix of distances between unvisited nodes
    unvisited_dist_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Calculate arithmetic mean of distances from each candidate to other unvisited nodes (Connectivity metric)
    # Low mean distance indicates the node is centrally located relative to the rest of the cluster.
    # We want to visit nodes that are centrally connected to avoid isolating peripheral nodes later.
    # We subtract the mean distance because a lower mean distance is desirable (better connectivity).
    # Thus, low mean_distance -> lower score -> higher priority.
    mean_distances = np.mean(unvisited_dist_matrix, axis=1)
    
    # Calculate destination proximity penalty
    # We want to penalize nodes that are far from the destination to ensure we end near it.
    # The penalty is scaled by 1/num_remaining to provide a linear ramp-up of importance as nodes remain.
    destination_distances = distance_matrix[unvisited_nodes, destination_node]
    destination_penalty = destination_distances / num_remaining
    
    # Calculate Nearest-Neighbor Regret
    # For each candidate, find the distance to the nearest and second-nearest OTHER unvisited nodes.
    # Regret = dist(second_nearest) - dist(nearest)
    # A high regret means there is a very close neighbor compared to others. 
    # We want to visit nodes that have close neighbors soon to "save" that edge before the neighbor is taken by someone else?
    # Actually, standard regret heuristics often suggest visiting nodes where the "opportunity cost" of not visiting the closest neighbor is high.
    # However, in the context of "next node to visit", if a node has a very close unvisited neighbor, visiting this node first allows us to potentially pick that neighbor next.
    # But typically, "regret" in TSP construction (like Cheapest Insertion) looks at how much the cost increases if we don't pick the best option.
    # Here, the prompt asks for: "penalize choices that eliminate high-value local connections".
    # If I pick a node that has a VERY close neighbor (high regret), I should prioritize it so I can pick that neighbor next? 
    # Or does it mean I should avoid picking a node if it doesn't have a unique close neighbor?
    # Let's look at the prompt: "penalize choices that eliminate high-value local connections".
    # If a node has a close neighbor, visiting it preserves the ability to visit that neighbor next.
    # If a node has NO close neighbor (low regret), visiting it is less critical in terms of local structure.
    # So we want to PREFER nodes with HIGH regret (unique close connections).
    # Therefore, we should SUBTRACT the regret from the score (since we minimize score).
    
    # Calculate k=2 nearest neighbors for each unvisited node within the unvisited set
    # We need distances to OTHER unvisited nodes. The diagonal is 0.
    # Replace diagonal with inf to ignore self-distance for sorting
    temp_dist_matrix = unvisited_dist_matrix.copy()
    np.fill_diagonal(temp_dist_matrix, np.inf)
    
    # Get the two smallest distances for each node
    # argsort might be slow for large arrays, but let's assume reasonable size or use partition
    # np.argpartition is faster for just getting k smallest
    k = 2
    # Get indices of the two smallest elements
    # Using argsort for simplicity and correctness in one line
    sorted_indices = np.argsort(temp_dist_matrix, axis=1)
    
    # Extract the distances for the 1st and 2nd nearest
    nearest_dists = temp_dist_matrix[np.arange(num_remaining), sorted_indices[:, 0]]
    second_nearest_dists = temp_dist_matrix[np.arange(num_remaining), sorted_indices[:, 1]]
    
    # Regret is the difference. If second is inf (only 1 other node), regret is inf? 
    # But we handled len==1 above. If len==2, second nearest is inf?
    # If num_remaining == 2, each node has only 1 other node. 
    # sorted_indices[:, 1] will be inf.
    # We should handle this edge case or let it propagate.
    # If regret is inf, score becomes -inf, which forces pick. This makes sense for last steps.
    
    # Enhance regret by incorporating the absolute distance to the nearest neighbor as a weighting factor.
    # Replace linear global mean baseline with robust median-of-normals estimate using IQR.
    # Flatten the distance matrix to get all pairwise distances between unvisited nodes
    flat_dists = temp_dist_matrix[temp_dist_matrix < np.inf]
    
    # Calculate median and IQR for robust scaling
    if len(flat_dists) > 0:
        median_dist = np.median(flat_dists)
        q1, q3 = np.percentile(flat_dists, [25, 75])
        iqr = q3 - q1
        # Robust scale estimate: Median Absolute Deviation (MAD) is more robust, 
        # but IQR is requested. Use IQR/1.34 as a robust estimate of std dev for normal distribution.
        # Or simply use the median as the baseline if IQR is zero or small.
        robust_scale = iqr / 1.34 if iqr > 0 else median_dist
        
        # Avoid division by zero
        if robust_scale == 0:
            robust_scale = 1e-9
            
        weight = 1.0 + nearest_dists / robust_scale
    else:
        # Fallback if no distances (e.g., single node, though handled above)
        weight = 1.0 * np.ones_like(nearest_dists)
        
    # Base regret calculation
    base_regret = second_nearest_dists - nearest_dists
    
    # Weighted regret
    weighted_regret = base_regret * weight
    
    # Coefficient for regret term
    regret_coeff = 0.2
    
    # Modified Bridge Metric with Destination Penalty and Regret:
    # Score = Immediate Distance - Alpha * Mean Distance to others - Destination Penalty - Regret_Coeff * Regret
    #
    # Alpha is now dynamically scaled to reduce connectivity bias in early steps (many nodes)
    # and increase it as the cluster shrinks, adhering to phase-dependent structural importance.
    # Base alpha is 0.7.
    # Refined scaling: use 0.5 instead of 1.0 to reduce aggressive early suppression.
    alpha = 0.7 * (1 - 0.5 / num_remaining)
    
    scores = current_distances - alpha * mean_distances - destination_penalty - regret_coeff * weighted_regret
    
    # Select the node with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

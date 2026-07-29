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
    # Handle edge case: no unvisited nodes
    if len(unvisited_nodes) == 0:
        # Return destination if it's valid, otherwise current_node
        return current_node
    
    # Handle edge case: only one unvisited node
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_unvisited = len(unvisited_nodes)
    N_total = distance_matrix.shape[0]
    
    # Get distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get distances from each unvisited node to the destination node
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Compute dynamic weight alpha using a sigmoid function.
    # This provides a smooth, non-linear transition.
    # When n_unvisited is large (early tour), alpha is small (prioritize immediate distance).
    # When n_unvisited is small (late tour), alpha is large (prioritize bridge-aware connectivity).
    f = n_unvisited / max(N_total, 1)
    
    # Extract the submatrix of distances between unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # --- MODIFICATION: Local "future cost" estimator using k-nearest neighbors ---
    
    # Determine k for k-NN. 
    # We want enough neighbors to get a stable estimate but few enough to be local.
    k_nn = min(10, n_unvisited - 1)
    if k_nn < 1:
        k_nn = 1
        
    mean_dists_to_remaining = np.zeros(n_unvisited)
    
    # For each candidate, find the k_nn nearest unvisited neighbors and compute their mean distance.
    for i in range(n_unvisited):
        row = sub_matrix[i, :]
        # Get indices sorted by distance
        sorted_indices = np.argsort(row)
        
        neighbors = []
        for idx in sorted_indices:
            if idx == i:
                continue
            neighbors.append(row[idx])
            if len(neighbors) == k_nn:
                break
        
        if len(neighbors) == 0:
            mean_dists_to_remaining[i] = 0
        else:
            mean_dists_to_remaining[i] = np.mean(neighbors)
    
    # --- End Modification ---
    
    # Compute global average distance of the local k-NN estimates for normalization
    global_mean_dist = np.mean(mean_dists_to_remaining)
    epsilon = 1e-10
    
    # Compute median distance of all unvisited edges (upper triangle of sub_matrix)
    # Flatten the upper triangle to get unique edge distances
    upper_tri = np.triu(sub_matrix, k=1)
    unvisited_edge_dists = upper_tri[upper_tri > 0]
    
    if len(unvisited_edge_dists) > 0:
        median_unvisited_dist = np.median(unvisited_edge_dists)
    else:
        median_unvisited_dist = 1.0
        
    # Compute ratio of average current step distance to median unvisited edge distance
    avg_current_dist = np.mean(dist_from_current)
    ratio = avg_current_dist / (median_unvisited_dist + epsilon)
    
    # Dynamic steepness parameter scaling with this ratio (from Primary)
    # Higher ratio means current step is relatively long, so we need sharper transition
    k = 2.0 * np.log1p(ratio)
    
    # Global sigmoid alpha based on tour progress (from Requested Modification)
    alpha = 1.0 / (1.0 + np.exp(k * (f - 0.5)))
    
    # Normalize bridge-aware mean distances by global unvisited density
    normalized_mean_dists = mean_dists_to_remaining / (global_mean_dist + epsilon)
    
    # --- MODIFICATION: Forward Progress Score ---
    # Replace dist_to_dest in harmonic core with forward progress score.
    # Progress = dist_from_current - dist_to_dest.
    # We want to reward progress, so we want a term that is SMALL when progress is GOOD (large positive).
    # Score = 2.0 * dist_to_dest - dist_from_current.
    # This score decreases as progress increases.
    
    modified_dest_term = 2.0 * dist_to_dest - dist_from_current
    
    # Ensure positivity for harmonic mean
    min_modified = np.min(modified_dest_term)
    shift = max(0.0, -min_modified) + epsilon
    positive_dest_term = modified_dest_term + shift
    
    # Calculate harmonic core with the new destination term
    harmonic_core = 2.0 * positive_dest_term * mean_dists_to_remaining / (positive_dest_term + mean_dists_to_remaining + epsilon)
    
    # --- NEW MODIFICATION: Detour Penalty based Gamma Scaling ---
    
    # Calculate distance from current node to destination
    dist_current_to_dest = distance_matrix[current_node, destination_node]
    
    # Compute detour penalty for each candidate
    # Detour = dist(current, i) + dist(i, dest) - dist(current, dest)
    # We only care about positive detours (inefficiency)
    raw_detour = dist_from_current + dist_to_dest - dist_current_to_dest
    detour = np.maximum(0.0, raw_detour)
    
    # Normalize detour by the maximum possible detour or a scale factor to keep beta effective
    # Using the max detour among candidates for normalization
    max_detour = np.max(detour)
    if max_detour > epsilon:
        normalized_detour = detour / max_detour
    else:
        normalized_detour = np.zeros(n_unvisited)
    
    # Base gamma scales inversely with n_unvisited
    gamma_base = 1.0 / max(n_unvisited, 1)
    
    # Dynamic scaling factor for gamma based on detour
    # beta controls the strength of the detour penalty
    beta = 1.0
    gamma_dynamic = gamma_base * (1.0 + beta * normalized_detour)
    
    # Normalize destination distances for the additive penalty term
    global_dist_to_dest = np.mean(dist_to_dest)
    normalized_dist_to_dest = dist_to_dest / (global_dist_to_dest + epsilon)
    
    # Additive penalty structure with zero-centered convex combination
    # Gamma is now dynamic per candidate
    scale_factor = 1.0 + alpha * (normalized_mean_dists - 1.0) + gamma_dynamic * (normalized_dist_to_dest - 1.0)
    
    dynamic_denominator = harmonic_core * scale_factor
    
    # Compute the ratio: distance from current / dynamic_denominator
    # We want to minimize this ratio: prefer nodes that are close to current 
    # and also have good connectivity/progress toward destination (high denominator)
    scores = dist_from_current / (dynamic_denominator + epsilon)
    
    # Select the node with the minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

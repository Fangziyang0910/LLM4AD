import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray, prizes: np.ndarray, remaining_budget: float) -> int:
    """
    Design a novel constructive heuristic for the Orienteering Problem.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the route destination node.
    unvisited_nodes: Array of feasible unvisited node IDs. Visiting one of these nodes still leaves enough budget to return to the destination.
    distance_matrix: Pairwise Euclidean distance matrix of all nodes.
    prizes: Prize values of all nodes. The depot prize is 0.
    remaining_budget: Remaining travel budget before selecting the next node.

    Return:
    ID of the next node to visit.
    """
    
    if len(unvisited_nodes) == 0:
        return destination_node
        
    # Get the subset of unvisited nodes
    candidates = unvisited_nodes
    
    # Calculate distances from current node to each candidate
    dists_to_candidates = distance_matrix[current_node, candidates]
    
    # Calculate distances from each candidate to the destination (to check return feasibility)
    dists_to_dest = distance_matrix[candidates, destination_node]
    
    # Total cost to visit candidate and return to destination
    total_costs = dists_to_candidates + dists_to_dest
    
    # Filter candidates that are feasible (total cost <= remaining budget)
    # Note: The problem statement says unvisited_nodes are already feasible, 
    # but we double-check to be safe.
    feasible_mask = total_costs <= remaining_budget
    
    # If no feasible candidates, return to destination
    if not np.any(feasible_mask):
        return destination_node
        
    feasible_candidates = candidates[feasible_mask]
    if len(feasible_candidates) == 0:
        return destination_node
        
    # Calculate prize-to-distance ratio for feasible candidates
    # Using distance from current node as the primary metric for "effort"
    # Ratio = Prize / Distance
    candidate_prizes = prizes[feasible_candidates]
    candidate_dists = dists_to_candidates[feasible_mask]
    
    # Avoid division by zero
    epsilon = 1e-6
    ratios = candidate_prizes / (candidate_dists + epsilon)
    
    # Filter to top 30% of the maximum ratio for sparsified subset
    if len(ratios) == 0:
        return destination_node
        
    max_ratio = np.max(ratios)
    if max_ratio == 0:
        # If all ratios are 0, just pick the closest feasible node
        best_index = np.argmin(candidate_dists)
        return feasible_candidates[best_index]

    threshold = max_ratio * 0.3 # Top 30%
    
    # High-efficiency subset: nodes with ratio >= 30% of the max ratio
    high_eff_mask = ratios >= threshold
    high_eff_candidates = feasible_candidates[high_eff_mask]
    
    # If filtering leaves no nodes (e.g., threshold too high), fallback to just feasible candidates
    if len(high_eff_candidates) == 0:
        high_eff_candidates = feasible_candidates
        high_eff_ratios = ratios
        # Use distance matrix rows for these candidates
        high_eff_dist_vectors = distance_matrix[high_eff_candidates]
    else:
        high_eff_ratios = ratios[high_eff_mask]
        # Use distance matrix rows for these candidates
        high_eff_dist_vectors = distance_matrix[high_eff_candidates]

    # If only 1 node left in subset, pick it
    if len(high_eff_candidates) == 1:
        return high_eff_candidates[0]
        
    # Calculate local density score for each node in the high-efficiency subset
    # We define a radius R based on the average pairwise distance among the high-efficiency candidates
    # To avoid O(N^2) for large sets, we can use a heuristic or just compute full matrix if N is small
    # Since we sparsified to top 30%, N should be manageable.
    
    n_high = len(high_eff_candidates)
    
    # Compute pairwise distances between high-efficiency candidates
    # We use the distance_matrix directly. 
    # dist_matrix_subset[i, j] is distance between high_eff_candidates[i] and high_eff_candidates[j]
    dist_matrix_subset = distance_matrix[high_eff_candidates][:, high_eff_candidates]
    
    # Calculate average distance among high-efficiency nodes to define radius
    # Exclude diagonal
    upper_tri_indices = np.triu_indices(n_high, k=1)
    if len(upper_tri_indices[0]) > 0:
        avg_dist = np.mean(dist_matrix_subset[upper_tri_indices])
    else:
        avg_dist = 0.0
        
    # Define radius as average distance. 
    # If avg_dist is 0 (all nodes same location?), set to a small positive number or use a fraction of max dist
    if avg_dist == 0:
        # Fallback: use max distance between any pair * 0.5
        max_dist = np.max(dist_matrix_subset)
        radius = max_dist * 0.5 if max_dist > 0 else 1.0
    else:
        radius = avg_dist
        
    # Count neighbors within radius for each node
    # Density score[i] = sum(1 for j != i if dist(i,j) <= radius)
    # We can use broadcasting or matrix operations
    # diff = dist_matrix_subset <= radius
    # np.fill_diagonal(diff, False)
    # density_scores = np.sum(diff, axis=1)
    
    # More efficient vectorized approach
    # Create a mask of distances <= radius
    close_mask = dist_matrix_subset <= radius
    # Exclude self-loops
    np.fill_diagonal(close_mask, False)
    
    density_scores = np.sum(close_mask, axis=1)
    
    # Combine efficiency ratio and density score
    # Normalize density scores to be comparable with ratios? 
    # Ratios can be large or small. Density is an integer count.
    # Let's scale density by a factor, e.g., max_ratio / (max_density + 1) or just add directly if scales are similar.
    # A safer approach: score = ratio + alpha * density_score
    # Let alpha be such that max density contribution is comparable to ratio variation.
    
    max_density = np.max(density_scores)
    if max_density == 0:
        # All isolated, just pick max ratio
        best_idx = np.argmax(high_eff_ratios)
        return high_eff_candidates[best_idx]
        
    # Normalize density to [0, 1] relative to max density in this subset
    norm_density = density_scores / max_density
    
    # Normalize ratios to [0, 1] relative to max ratio in this subset
    # Note: high_eff_ratios are already >= 0.3 * max_global_ratio
    min_eff_ratio = np.min(high_eff_ratios)
    max_eff_ratio = np.max(high_eff_ratios)
    
    if max_eff_ratio - min_eff_ratio > 1e-6:
        norm_ratios = (high_eff_ratios - min_eff_ratio) / (max_eff_ratio - min_eff_ratio)
    else:
        norm_ratios = np.ones_like(high_eff_ratios)
        
    # Combined score: weighted sum. 
    # Give slight preference to efficiency, but density is a strong boost.
    # Score = 0.6 * norm_ratio + 0.4 * norm_density
    combined_scores = 0.6 * norm_ratios + 0.4 * norm_density
    
    # Select node with max combined score
    best_local_index = np.argmax(combined_scores)
    next_node = high_eff_candidates[best_local_index]
    
    return next_node

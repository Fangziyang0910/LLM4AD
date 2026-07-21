
import numpy as np
def update_edge_distance(edge_distance: np.ndarray, local_opt_tour: np.ndarray, edge_n_used: np.ndarray) -> np.ndarray:
    """
    Design a novel algorithm to update the distance matrix.

    Args:
    edge_distance: A matrix of the distance.
    local_opt_tour: An array of the local optimal tour of IDs.
    edge_n_used: A matrix of the number of each edge used during permutation.

    Return:
    updated_edge_distance: A matrix of the updated distance.
    """
    updated_edge_distance = edge_distance.copy()
    n = len(local_opt_tour)
    
    if n == 0:
        return updated_edge_distance

    # Identify edges in the current tour
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.append((u, v))
        
    # Calculate global statistics for normalization
    # Avoid division by zero
    max_usage_global = np.max(edge_n_used)
    if max_usage_global < 1e-8:
        max_usage_global = 1.0
        
    # Global min distance for scale invariance
    valid_dists = edge_distance[edge_distance > 1e-8]
    if len(valid_dists) > 0:
        global_min_dist = np.min(valid_dists)
    else:
        global_min_dist = 1.0
        
    if global_min_dist < 1e-8:
        global_min_dist = 1.0

    # Calculate a global reference for "goodness" of alternatives
    # We compute the minimum alternative distance for every node to establish a baseline gap
    # This helps in normalizing the local gap against the global distribution of gaps
    
    # Precompute min alternative for each node (excluding self)
    # To save time in the loop, we can compute this once or approximate it
    # However, for precision, we will compute the specific alternative for each tour edge on the fly
    # but use global stats for normalization.
    
    # Let's compute the mean absolute gap across the graph for normalization
    # gap(u, v) = dist(u, v) - min(dist(u, k) for k != u, v)
    # This is expensive to compute for all edges, so we sample or use a simpler heuristic.
    # Instead, let's use the mean of all positive distances as a scale reference.
    mean_dist_global = np.mean(valid_dists) if len(valid_dists) > 0 else 1.0
    
    # Parameters tuned to combine strengths of Alg 1, 2, 3
    sigmoid_steepness = 5.0      # Steeper sigmoid for more decisive action
    usage_decay_rate = 1.5       # Moderate decay to protect used edges but allow escape
    penalty_max = 4.0            # Cap the penalty to prevent divergence
    
    for u, v in tour_edges:
        current_dist = edge_distance[u, v]
        current_usage = edge_n_used[u, v]
        
        # Find the minimum distance from u to any other node k, where k != v and k != u
        dists_from_u = edge_distance[u, :]
        
        # Mask out the destination v and the source u itself
        alternative_mask = np.ones(n, dtype=bool)
        alternative_mask[v] = False
        alternative_mask[u] = False
        
        if np.any(alternative_mask):
            alt_dists = dists_from_u[alternative_mask]
            min_alt_dist = np.min(alt_dists)
            
            # Calculate the absolute difference (gap)
            diff = current_dist - min_alt_dist
            
            # Normalize the difference by the current distance to make it scale-invariant
            # If current_dist is very small, relative diff can be noisy, so clamp or use absolute
            if current_dist > 1e-8:
                relative_diff = diff / current_dist
            else:
                relative_diff = 0.0
                
            # Also consider the gap relative to the global minimum distance for context
            # This helps if the local alternative is good globally, even if not locally best relative to a huge current edge
            relative_global_gap = diff / global_min_dist if diff > 0 else 0.0
            
            # Combine relative differences: 
            # We want to penalize if the edge is significantly worse than the best local alternative
            # AND if that alternative is globally competitive.
            # Using a weighted combination or just the local relative diff is usually sufficient for TSP.
            # Let's stick to relative_diff but scale it by a factor derived from global stats if needed.
            
            # Threshold for activation: 
            # Alg 3 used 0.03. Alg 1 used 0.1. 
            # A sigmoid centered at 0.05 with steepness 5 transitions from 0.5 to ~0.99 between 0.05 and 0.15.
            activation_threshold = 0.05 
            
            # Sigmoidal function for relative cost gap
            # This maps the relative difference to a [0, 1] factor
            # If relative_diff < threshold, penalty is very small.
            cost_gap_factor = 1.0 / (1.0 + np.exp(-sigmoid_steepness * (relative_diff - activation_threshold)))
            
            # Usage-based weight using exponential decay
            # Normalize usage to [0, 1]
            usage_norm = current_usage / max_usage_global
            
            # Exponential decay: exp(-k * usage_norm)
            # When usage is 0, factor is 1 (max penalty). 
            # When usage is high, factor approaches 0 (protected).
            usage_factor = np.exp(-usage_decay_rate * usage_norm)
            
            # Final penalty calculation
            # Scale by the actual difference magnitude to ensure larger gaps get larger penalties
            # But cap it to prevent instability
            raw_penalty = penalty_max * cost_gap_factor * usage_factor
            
            # We apply the penalty to the distance.
            # To make it proportional to the gap size, we can multiply by (diff / global_min_dist)
            # However, cost_gap_factor already captures the 'badness'.
            # Let's add a linear component for the gap size to ensure the penalty scales with the error magnitude.
            # gap_scale = diff / (global_min_dist + 1e-8)
            # final_penalty = raw_penalty * (1.0 + gap_scale)
            
            # Simpler approach: Just add the raw_penalty. 
            # The sigmoid ensures we only penalize when relative_diff is high.
            # The usage_factor protects used edges.
            
            updated_edge_distance[u, v] += raw_penalty
            updated_edge_distance[v, u] += raw_penalty
            
    return updated_edge_distance

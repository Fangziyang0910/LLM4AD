import numpy as np
from collections import defaultdict

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
    n = edge_distance.shape[0]
    if n == 0:
        return edge_distance.copy()
    
    # Create a copy of the distance matrix to avoid modifying the original
    updated_distance = edge_distance.copy()
    
    # Parameters for the heuristic
    base_scale = 0.5      # Base multiplier for the penalty magnitude
    usage_alpha = 0.1     # Coefficient for usage damping (used in formula 1/(1+alpha*usage))
    var_scale = 1.0       # Scale factor for variance modulation
    eps = 1e-8            # Small constant to avoid division by zero
    k_best = 5            # Number of best alternatives to consider
    decay_rate = 0.95     # Exponential decay factor for edge_n_used (applied to simulate recency)
    
    num_nodes_in_tour = len(local_opt_tour)
    
    # --- Usage-Frequency Decay Mechanism ---
    # Apply exponential decay to the usage counts.
    # This reduces the influence of older uses, so that 'edge_n_used' reflects
    # recent activity more strongly than distant history.
    decayed_edge_n_used = edge_n_used * decay_rate
    
    # Identify unique nodes in the tour to process
    unique_nodes = np.unique(local_opt_tour)
    
    # Pre-compute best alternatives for each unique node
    # best_alt_dist[u] will store the smallest edge distance from u to any node v 
    # such that v is not a direct neighbor of u in the local_opt_tour.
    
    # First, map each node in the tour to its neighbors in the tour
    tour_neighbors = defaultdict(set)
    for i in range(num_nodes_in_tour):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % num_nodes_in_tour]
        tour_neighbors[int(u)].add(int(v))
        tour_neighbors[int(v)].add(int(u))
        
    best_alt_cache = {}
    
    for u in unique_nodes:
        u_int = int(u)
        
        # Get neighbors in the tour
        neighbors_in_tour = tour_neighbors[u_int]
        
        # We need the smallest edge from u to any node NOT in neighbors_in_tour and != u
        # Get the row for u
        row = edge_distance[u_int]
        
        # Mask out u itself and its tour neighbors
        # Strategy: Use np.argpartition to find the smallest values, then filter.
        
        # If n is small, argpartition is fine. If n is large, this is efficient.
        # We need at least k_best candidates that are not neighbors.
        # Let's fetch the top (k_best + len(neighbors_in_tour)) smallest distances to ensure we find k_best valid ones.
        num_to_fetch = k_best + len(neighbors_in_tour) + 1 # +1 for self
        
        if num_to_fetch >= n:
            num_to_fetch = n
            
        # Get indices of the smallest elements
        # np.argpartition returns indices such that the element at partition index is in sorted position
        # and all elements to left are <= and all to right are >=
        if num_to_fetch > 0:
            partition_indices = np.argpartition(row, num_to_fetch)[:num_to_fetch]
        else:
            partition_indices = []
            
        # Filter out invalid indices (self and tour neighbors)
        valid_alts = []
        for idx in partition_indices:
            if idx == u_int:
                continue
            if idx in neighbors_in_tour:
                continue
            valid_alts.append((row[idx], idx))
            
        # Sort the valid alternatives to get the true minimum
        valid_alts.sort(key=lambda x: x[0])
        
        if valid_alts:
            best_alt_cache[u_int] = valid_alts[0][0] # Store the min distance
        else:
            # If no valid alternative (e.g., tour covers all nodes and n is small),
            # we might fallback to global min excluding neighbors, but for large N this is rare.
            # If truly no alternative, we set a flag or a high value.
            # For robustness, let's do a full scan if the partition failed to find enough, 
            # though with k_best=5 and typical tours, this is unlikely unless n is very small.
            # Simple fallback: find min excluding neighbors via full boolean mask
            mask = np.ones(n, dtype=bool)
            mask[u_int] = False
            for nb in neighbors_in_tour:
                mask[int(nb)] = False
                
            if np.any(mask):
                best_alt_cache[u_int] = np.min(row[mask])
            else:
                best_alt_cache[u_int] = float('inf') # No alternative

    # Now iterate through the tour edges and apply penalty
    for i in range(num_nodes_in_tour):
        u = int(local_opt_tour[i])
        v = int(local_opt_tour[(i + 1) % num_nodes_in_tour])
        
        # Current edge distance
        current_dist = edge_distance[u, v]
        
        # Get best alternative distances for u and v from cache
        min_alt_dist_u = best_alt_cache.get(u, float('inf'))
        min_alt_dist_v = best_alt_cache.get(v, float('inf'))
        
        # Use the better (smaller) alternative distance from either endpoint
        min_alt_dist = min(min_alt_dist_u, min_alt_dist_v)
        
        # Calculate deviation ratio
        if min_alt_dist > eps and current_dist > min_alt_dist:
            deviation_ratio = current_dist / min_alt_dist
        else:
            deviation_ratio = 0.0
            
        # Get historical usage (now decayed to reflect recency)
        usage_count = decayed_edge_n_used[u, v]
        
        # Calculate modulation factor based on decayed usage and inverse distance
        # Standard damping: 1 / (1 + alpha * usage)
        usage_damping = 1.0 / (1.0 + usage_alpha * usage_count)
        
        # Inverse distance weighting modulation
        # Dividing the damping term by current edge distance
        base_modulation = usage_damping / (current_dist + eps)
        
        # --- Diversity-driven adaptive penalty with local density variance ---
        
        # Calculate local density variance for u
        # Neighbors of u in the tour
        neighbors_u = tour_neighbors[u]
        dists_u = []
        for nb in neighbors_u:
            dists_u.append(edge_distance[u, nb])
        
        # Calculate local density variance for v
        # Neighbors of v in the tour
        neighbors_v = tour_neighbors[v]
        dists_v = []
        for nb in neighbors_v:
            dists_v.append(edge_distance[v, nb])
            
        # Combine the distances from both endpoints to get a local cluster variance
        # If a node has only 1 neighbor in the tour (start/end of path if not closed, but TSP is closed),
        # std dev will be 0. For TSP, every node has exactly 2 neighbors in the tour.
        all_local_dists = dists_u + dists_v
        
        if len(all_local_dists) > 1:
            local_std_dev = np.std(all_local_dists)
        else:
            local_std_dev = 0.0
            
        # Normalize the standard deviation. 
        # We can scale it by the mean distance of the local cluster or just use it directly with a scale factor.
        # To prevent explosion, we clamp or normalize. 
        # Let's use a simple normalization: if mean is 0, use eps.
        local_mean_dist = np.mean(all_local_dists) if len(all_local_dists) > 0 else eps
        normalized_var_factor = local_std_dev / (local_mean_dist + eps)
        
        # The variance term should amplify the penalty. 
        # If variance is high, the edge is likely an outlier or part of an irregular cluster.
        # We add 1.0 to ensure a baseline even if var is 0, and scale by var_scale.
        variance_modulation = 1.0 + var_scale * normalized_var_factor
        
        # Final modulation combines base modulation and variance modulation
        final_modulation = base_modulation * variance_modulation
        
        # Calculate penalty
        # Penalty is added to the distance to make it less attractive.
        # Formula derived: penalty = base_scale * deviation_ratio * final_modulation
        penalty = base_scale * deviation_ratio * final_modulation
        
        # Add penalty to the edge distance
        updated_distance[u, v] += penalty
        updated_distance[v, u] += penalty
        
    # Ensure symmetry (already handled by updating both [u,v] and [v,u])
    # Ensure non-negativity
    updated_distance = np.maximum(updated_distance, 0)
    
    return updated_distance

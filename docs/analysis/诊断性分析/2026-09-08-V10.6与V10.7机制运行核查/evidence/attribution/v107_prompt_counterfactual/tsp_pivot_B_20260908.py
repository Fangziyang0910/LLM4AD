import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Selects the next node using a 2-step look-ahead heuristic with dynamic return bias
    and isolation penalties.
    """
    n_remaining = len(unvisited_nodes)
    
    if n_remaining == 0:
        return destination_node
        
    if n_remaining == 1:
        return int(unvisited_nodes[0])

    n_total = distance_matrix.shape[0]
    
    # 1. Extract basic distances
    # d_cur: distance from current node to each candidate
    d_cur = distance_matrix[current_node, unvisited_nodes]
    # d_dest: distance from each candidate to the destination (start node)
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 2. Construct submatrix of distances among unvisited nodes
    # This represents d(i, j) for all i, j in unvisited
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].copy()
    # Set diagonal to infinity to prevent self-loops in look-ahead
    np.fill_diagonal(sub_matrix, np.inf)
    
    # 3. Calculate 2-step Look-Ahead Cost
    # For each candidate i, the look-ahead cost is the minimum cost of:
    # Moving from i to some other unvisited node j, then from j to destination.
    # Cost(i) = min_j ( d(i, j) + d(j, dest) )
    # We can compute this efficiently using matrix addition and min.
    # sub_matrix[i, j] is d(i, j)
    # d_dest[j] is d(j, dest)
    # We want min over j for each i.
    # Note: If we pick i, the next node j must be different from i. 
    # Since diagonal is inf, min_j will naturally pick j != i.
    
    # However, we must consider if the "best next step" after i is actually the destination itself.
    # The problem statement says we visit each node once.
    # So if we are at i, and there are other unvisited nodes, we MUST visit one of them next.
    # Only when no other unvisited nodes remain do we go to destination.
    # So the look-ahead term min_j (d(i,j) + d(j,dest)) is valid for n_remaining > 1.
    
    # Let's calculate the transition cost matrix: T[i, j] = d(i, j) + d(j, dest)
    transition_cost = sub_matrix + d_dest[np.newaxis, :]
    
    # Look-ahead cost for each candidate i: min over j != i
    look_ahead_cost = np.min(transition_cost, axis=1)
    
    # 4. Calculate Isolation Penalty
    # Isolation measures how far a node is from the rest of the cluster of unvisited nodes.
    # High isolation suggests a node on the "periphery", which might be good to visit early
    # to keep the tour compact, or bad if it creates a long detour.
    # Let's use the average distance to other unvisited nodes as a proxy for isolation.
    if n_remaining > 1:
        # Sum of distances to all other unvisited nodes
        iso_sum = np.sum(sub_matrix, axis=1)
        # Average isolation
        iso_avg = iso_sum / (n_remaining - 1)
    else:
        iso_avg = np.zeros(n_remaining)
    
    # Normalize isolation to have comparable scale to distances
    # We can divide by the max average distance or just use it raw if distances are similar.
    # Let's normalize by the mean of d_cur to keep weights stable.
    scale = np.mean(d_cur) if np.mean(d_cur) > 0 else 1.0
    
    # 5. Dynamic Weights based on Progress
    # progress: 0 at start, 1 at end
    if n_total > 1:
        progress = (n_total - n_remaining) / (n_total - 1)
    else:
        progress = 1.0
        
    # Return Bias: Increases as we get closer to the end to ensure we can return to start.
    # Use a quadratic ramp to make it stronger later.
    w_return = 0.1 + 0.5 * (progress ** 2)
    
    # Look-ahead Weight: Constant or slightly decreasing?
    # Look-ahead helps plan the path, important throughout.
    w_look_ahead = 0.4
    
    # Isolation Weight: 
    # Early in the tour (low progress), we might want to visit isolated nodes first to "clean up" the periphery?
    # Or should we visit dense nodes to keep the path short?
    # Usually, nearest neighbor works by keeping things local.
    # High isolation means far from others. Visiting a highly isolated node early might stretch the tour.
    # So we want to PENALIZE high isolation early? Or prefer low isolation?
    # Let's prefer nodes that are NOT too isolated from the remaining set, to keep the tour compact.
    # So score includes + w_iso * iso_avg.
    # Weight decreases as tour progresses because the "remaining set" is smaller and less relevant for global compactness.
    w_iso = 0.2 * (1.0 - progress)
    
    # Immediate Distance Weight: Base cost
    w_cur = 1.0
    
    # 6. Compute Composite Score
    scores = (w_cur * d_cur + 
              w_look_ahead * look_ahead_cost + 
              w_return * d_dest + 
              w_iso * (iso_avg / scale) * scale) # Multiply back by scale to make it comparable to d_cur?
    
    # Let's refine the scaling. d_cur, look_ahead, d_dest are all raw distances.
    # iso_avg is also a raw distance.
    # So we don't need to normalize if we want weights to be direct multipliers on distance.
    # However, if d_cur is small and iso is large, iso might dominate.
    # Let's just use raw values with tuned weights.
    
    scores = (1.0 * d_cur + 
              0.3 * look_ahead_cost + 
              w_return * d_dest + 
              0.2 * (1.0 - progress) * iso_avg)
              
    # Handle potential numerical issues
    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
        # Fallback to nearest neighbor if issues arise
        scores = d_cur
        
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
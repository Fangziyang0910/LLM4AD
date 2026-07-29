import numpy as np
def compute_mst_weight(nodes_subset, dist_mat):
    """
    Compute the weight of the Minimum Spanning Tree (MST) of a subset of nodes
    using Prim's algorithm. 
    Assumes nodes_subset is a 1D array of node IDs.
    """
    if len(nodes_subset) <= 1:
        return 0.0
    
    n = len(nodes_subset)
    
    # Prim's algorithm
    # Key values to pick minimum weight edge from cut
    key = np.full(n, float('inf'))
    in_mst = np.zeros(n, dtype=bool)
    
    # Start with node 0
    key[0] = 0.0
    
    mst_weight = 0.0
    
    for _ in range(n):
        # Find the node with min key not in MST
        u = -1
        min_val = float('inf')
        for i in range(n):
            if not in_mst[i] and key[i] < min_val:
                min_val = key[i]
                u = i
        
        if u == -1:
            break
            
        in_mst[u] = True
        mst_weight += min_val
        
        # Update keys of adjacent vertices
        node_u_id = nodes_subset[u]
        
        # We need distances from node_u_id to all other nodes in nodes_subset
        # Vectorized lookup
        dists = dist_mat[node_u_id, nodes_subset]
        
        for i in range(n):
            if not in_mst[i]:
                d = dists[i]
                if d < key[i]:
                    key[i] = d

    return mst_weight


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
        # No unvisited nodes left, return to destination
        return destination_node
    
    if len(unvisited_nodes) == 1:
        # Only one unvisited node left
        return unvisited_nodes[0]
    
    # Vectorized lookup of distances from current node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get indices of unvisited nodes sorted by distance from current node
    sorted_indices = np.argsort(dists_from_current)
    sorted_candidates = unvisited_nodes[sorted_indices]
    
    # Adaptive dynamic k selection
    n_total_unvisited = len(unvisited_nodes)
    unvisited_fraction = n_total_unvisited / max(n_total_unvisited, 1) 
    k_pool = max(3, int(np.ceil(10 * unvisited_fraction)))
    # Cap k to avoid evaluating too many if n is large
    k_pool = min(k_pool, n_total_unvisited)
    
    candidates_to_evaluate = []
    for i in range(k_pool):
        candidates_to_evaluate.append(sorted_candidates[i])
    
    best_node = None
    best_score = float('inf')
    
    # Dynamic weight for lookahead cost: increases as remaining nodes decrease
    n_remaining = len(unvisited_nodes)
    lookahead_weight = 1.0 + 5.0 * (1.0 / (n_remaining + 1.0))

    # Distance to the absolute nearest neighbor for deviation calculation
    dist_to_nearest = dists_from_current[0]
    
    # Calculate initial global MST weight for dynamic scaling of MST penalty
    try:
        initial_mst_weight = compute_mst_weight(unvisited_nodes, distance_matrix)
    except:
        initial_mst_weight = 0.0

    # Weights for penalty terms (Static baselines for non-MST terms)
    weight_deviation = 1.0
    weight_spread = 0.2
    weight_lookahead_2opt = 0.5
    weight_angular = 0.4  
    weight_dest_proximity = 1.0 
    
    # Precompute distances among all unvisited nodes for centroid and MST calculations
    try:
        sub_mat = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        # Sum of distances from each node to all others (used for angular heuristic)
        centroid_dist_sum = np.sum(sub_mat, axis=1)
        
        # Compute Peripherality Score for Convex Hull Preference
        # Score = Average Distance to others / Max Pairwise Distance in cluster
        avg_dists = np.mean(sub_mat, axis=1)
        max_pairwise_dist = np.max(sub_mat)
        
        if max_pairwise_dist > 1e-9:
            peripherality_scores = avg_dists / max_pairwise_dist
        else:
            peripherality_scores = np.zeros(len(unvisited_nodes))
            
    except:
        centroid_dist_sum = np.zeros(len(unvisited_nodes))
        sub_mat = None
        peripherality_scores = np.zeros(len(unvisited_nodes))

    for idx, candidate in enumerate(candidates_to_evaluate):
        # Step 1: Distance from current to candidate
        dist_to_candidate = distance_matrix[current_node, candidate]
        
        # Identify remaining unvisited nodes after visiting candidate
        # Efficiently create remaining array using mask-based filtering
        mask = np.ones(len(unvisited_nodes), dtype=bool)
        mask[unvisited_nodes == candidate] = False
        remaining = unvisited_nodes[mask]
        
        # Get index of candidate in unvisited_nodes for lookup
        try:
            cand_idx = np.where(unvisited_nodes == candidate)[0][0]
        except:
            cand_idx = 0
            
        # --- Angular Sector Coverage Heuristic ---
        max_centroid_dist = np.max(centroid_dist_sum) if len(centroid_dist_sum) > 0 else 0.0
        min_centroid_dist = np.min(centroid_dist_sum) if len(centroid_dist_sum) > 0 else 0.0
        
        range_centroid = max_centroid_dist - min_centroid_dist
        if range_centroid < 1e-9:
            range_centroid = 1.0
            
        candidate_central_sum = centroid_dist_sum[cand_idx]
        
        # Normalize: 0 = most central, 1 = most peripheral
        norm_centrality = (candidate_central_sum - min_centroid_dist) / range_centroid
        
        # We want to reward peripheral nodes (high norm_centrality).
        # So the penalty should be high for central nodes (low norm_centrality).
        angular_penalty = (1.0 - norm_centrality) * weight_angular * dist_to_nearest
        
        # --- Convex Hull Boundary Preference ---
        candidate_peripherality = peripherality_scores[cand_idx]
        # Penalty is high for central nodes (low peripherality), low for boundary nodes (high peripherality).
        # We will apply a dynamic weight to this in the final combination, but calculate base penalty here
        convex_hull_base_penalty = (1.0 - candidate_peripherality) * dist_to_nearest

        # --- Destination Proximity Bias ---
        dest_proximity_penalty = 0.0
        if len(remaining) > 0:
            dist_to_dest = distance_matrix[candidate, destination_node]
            dists_to_remaining = distance_matrix[candidate, remaining]
            dist_to_nearest_remaining = np.min(dists_to_remaining)
            
            if dist_to_nearest_remaining > 1e-9:
                ratio = dist_to_dest / dist_to_nearest_remaining
                if ratio < 1.0:
                    dest_proximity_penalty = weight_dest_proximity * (1.0 - ratio) * dist_to_nearest
                else:
                    dest_proximity_penalty = 0.0
            else:
                dest_proximity_penalty = 0.0
        else:
            dest_proximity_penalty = 0.0

        if len(remaining) == 0:
            # Candidate is the last unvisited node, go directly to destination
            dist_to_dest = distance_matrix[candidate, destination_node]
            total_score = dist_to_candidate + dist_to_dest + angular_penalty + convex_hull_base_penalty * weight_angular + dest_proximity_penalty
        else:
            # Step 2: Greedy estimate of path through remaining nodes back to destination
            
            # Start from candidate, greedily visit remaining nodes, then go to destination
            current = candidate
            greedy_path_cost = 0
            
            temp_remaining = remaining.copy()
            
            while len(temp_remaining) > 0:
                dists_to_remaining = distance_matrix[current, temp_remaining]
                min_idx = np.argmin(dists_to_remaining)
                min_node = temp_remaining[min_idx]
                min_dist = dists_to_remaining[min_idx]
                
                greedy_path_cost += min_dist
                current = min_node
                temp_remaining = np.delete(temp_remaining, min_idx)
            
            dist_to_dest = distance_matrix[current, destination_node]
            
            # Calculate spread for existing logic
            dists_from_candidate_to_remaining = distance_matrix[candidate, remaining]
            spread = np.std(dists_from_candidate_to_remaining)
            
            if dist_to_nearest > 1e-9:
                deviation_ratio = (dist_to_candidate - dist_to_nearest) / dist_to_nearest
            else:
                deviation_ratio = 0.0
            
            # Heuristic Term: MST Disruption Penalty with Dynamic Weighting
            # Compute MST of the REMAINING nodes to penalize scattering
            try:
                remaining_mst_weight = compute_mst_weight(remaining, distance_matrix)
            except:
                remaining_mst_weight = 0.0
                
            # Dynamic MST penalty weighting strategy
            # Scale the penalty term by the ratio of remaining to initial MST weight
            if initial_mst_weight > 1e-9:
                mst_ratio = remaining_mst_weight / initial_mst_weight
            else:
                mst_ratio = 0.0
            
            # Dynamic weights for MST and Convex Hull based on MST ratio
            # When mst_ratio is high (early tour, high residual cost), we penalize structural disruption more heavily
            # Base weights are 0.1 for MST and we use a multiplier for Convex Hull
            base_mst_weight = 0.1
            dynamic_mst_weight = base_mst_weight * (1.0 + mst_ratio)
            
            # Convex Hull weight scales similarly to enforce boundary following when structure is fragile
            base_ch_weight = 0.4 # Similar scale to angular
            dynamic_ch_weight = base_ch_weight * (1.0 + mst_ratio)

            mst_term = dynamic_mst_weight * remaining_mst_weight
            convex_hull_penalty = dynamic_ch_weight * convex_hull_base_penalty

            # Nearest-Neighbor Exit Penalty
            dists_from_candidate = distance_matrix[candidate, remaining]
            # Get indices of the two nearest neighbors
            if len(remaining) >= 2:
                sorted_remaining_indices = np.argsort(dists_from_candidate)
                nearest_dist_1 = dists_from_candidate[sorted_remaining_indices[0]]
                nearest_dist_2 = dists_from_candidate[sorted_remaining_indices[1]]
                avg_nearest_dist = (nearest_dist_1 + nearest_dist_2) / 2.0
            elif len(remaining) == 1:
                avg_nearest_dist = dists_from_candidate[0]
            else:
                avg_nearest_dist = 0.0
            
            # Apply decay factor to reduce impact as tour completes
            exit_penalty_weight = 0.5 # Base weight
            decay_factor = 1.0 / (n_remaining + 1.0)
            exit_penalty_term = exit_penalty_weight * decay_factor * avg_nearest_dist

            # Lookahead 2-Opt Swap Evaluation
            lookahead_2opt_penalty = 0.0
            if len(remaining) >= 1:
                # Find nearest neighbor of candidate among remaining
                dists_from_candidate = distance_matrix[candidate, remaining]
                nearest_idx = np.argmin(dists_from_candidate)
                nearest_node = remaining[nearest_idx]
                
                dist_candidate_to_nearest = dists_from_candidate[nearest_idx]
                dist_current_to_nearest = distance_matrix[current_node, nearest_node]
                
                # Current path segment: current -> candidate -> nearest
                path_through_candidate = dist_to_candidate + dist_candidate_to_nearest
                
                # Alternative path segment: current -> nearest -> candidate
                alt_cost = dist_current_to_nearest + dist_candidate_to_nearest
                
                # Incorporate MST-based structural disruption cost of the swapped nearest neighbor
                # We calculate the incremental MST cost of adding 'nearest_node' to the set of 
                # remaining nodes excluding 'nearest_node' itself.
                # effectively: MST(remaining) - MST(remaining \ {nearest_node}) is the cost contribution.
                # However, to penalize disruption, we look at how much 'nearest_node' contributes to the connectivity.
                # If we swap, 'nearest_node' becomes the new head. We want to ensure it doesn't isolate the rest.
                # A simple proxy for structural disruption is the incremental MST weight added by nearest_node.
                
                if len(remaining) > 1:
                    remaining_excluding_nearest = remaining[remaining != nearest_node]
                    try:
                        mst_excluding_nearest = compute_mst_weight(remaining_excluding_nearest, distance_matrix)
                    except:
                        mst_excluding_nearest = 0.0
                    
                    # The cost to connect nearest_node to the rest of the MST is roughly:
                    # remaining_mst_weight - mst_excluding_nearest
                    # This represents the "structural load" of nearest_node.
                    # We add this to the alternative cost to penalize choosing a node that heavily burdens the MST structure.
                    incremental_mst_cost = remaining_mst_weight - mst_excluding_nearest
                    
                    # Scale by dynamic weight to align with main MST penalty
                    structural_penalty = dynamic_mst_weight * incremental_mst_cost
                    
                    alt_cost += structural_penalty

                # Penalize if the chosen path is worse than the alternative configuration
                if path_through_candidate > alt_cost:
                    lookahead_2opt_penalty = path_through_candidate - alt_cost
                else:
                    lookahead_2opt_penalty = 0.0
            
            # Combine scores
            total_score = (dist_to_candidate + 
                           lookahead_weight * (greedy_path_cost + dist_to_dest) + 
                           weight_deviation * deviation_ratio * dist_to_nearest + 
                           weight_spread * spread +
                           mst_term +
                           exit_penalty_term +
                           weight_lookahead_2opt * lookahead_2opt_penalty +
                           angular_penalty +
                           dest_proximity_penalty +
                           convex_hull_penalty)
        
        if total_score < best_score:
            best_score = total_score
            best_node = candidate
    
    # Fallback if best_node was not set
    if best_node is None:
        best_node = sorted_candidates[0]
        
    return best_node

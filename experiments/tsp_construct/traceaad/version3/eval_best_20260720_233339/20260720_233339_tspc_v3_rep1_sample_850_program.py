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
        return unvisited_nodes[0]
    
    n_unvisited = len(unvisited_nodes)
    
    # --- Helper Functions ---

    def compute_1_median(nodes, dist_mat):
        """
        Finds the node in the set that minimizes the sum of distances to all other nodes in the set.
        """
        if len(nodes) == 0:
            return None, float('inf')
        if len(nodes) == 1:
            return nodes[0], 0.0
        
        try:
            dist_sub = dist_mat[np.ix_(nodes, nodes)]
        except Exception:
            return None, float('inf')
        
        sum_dists = np.sum(dist_sub, axis=1)
        median_idx = np.argmin(sum_dists)
        median_node = nodes[median_idx]
        median_cost = sum_dists[median_idx]
        
        return median_node, median_cost

    def calculate_cosine_similarity(prev_node, current_node, next_node, dist_mat):
        """
        Calculates the cosine similarity between the vector (prev->current) and (current->next).
        Uses Law of Cosines.
        
        Returns Trajectory Continuity (TC) in range [-1, 1].
        TC = 1 means straight path, TC = -1 means turn back.
        """
        if prev_node is None:
            return 0.5 # Neutral if no previous node
            
        d_prev_curr = dist_mat[prev_node, current_node]
        d_curr_next = dist_mat[current_node, next_node]
        d_prev_next = dist_mat[prev_node, next_node]
        
        if d_prev_curr == 0 or d_curr_next == 0:
            return 0.5

        denom = 2 * d_prev_curr * d_curr_next
        if denom == 0:
            return 0.5

        cos_angle = (d_prev_curr**2 + d_curr_next**2 - d_prev_next**2) / denom
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        tc = -cos_angle
        return tc

    def calculate_destination_alignment(current_node, candidate_node, destination_node, dist_mat):
        """
        Calculates the cosine similarity between the vector (current->destination) and (current->candidate).
        Returns Destination Alignment (DA) in range [-1, 1].
        DA = 1 means perfect alignment towards destination.
        """
        d_curr_dest = dist_mat[current_node, destination_node]
        d_curr_cand = dist_mat[current_node, candidate_node]
        d_cand_dest = dist_mat[candidate_node, destination_node]
        
        if d_curr_dest == 0 or d_curr_cand == 0:
            return 0.5

        denom = 2 * d_curr_dest * d_curr_cand
        if denom == 0:
            return 0.5

        cos_angle = (d_curr_dest**2 + d_curr_cand**2 - d_cand_dest**2) / denom
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        
        return cos_angle

    def compute_cluster_density_ratio_and_continuity(current, remaining_nodes, dist_mat, prev_node=None, target_node=None):
        """
        Calculates Cluster Density Ratio (CDR) and Trajectory Continuity (TC) for a specific target_node.
        """
        if len(remaining_nodes) < 1:
            return 0.0, 0.5 
            
        med_node, med_cost = compute_1_median(remaining_nodes, dist_mat)
        
        if med_node is None:
            return 0.0, 0.5
            
        avg_intra_dist = med_cost / len(remaining_nodes)
        
        dists_from_current = dist_mat[current, remaining_nodes]
        avg_inter_dist = np.mean(dists_from_current)
        
        if avg_inter_dist == 0:
            return 0.0, 0.5 
            
        cdr = avg_intra_dist / avg_inter_dist
        
        tc = calculate_cosine_similarity(prev_node, current, target_node, dist_mat)
        
        return cdr, tc

    def calculate_max_link_penalty(node, remaining_nodes, dist_mat):
        """
        Calculates a Max-Link Penalty score.
        High score = Bad (Node is far from the single most distant remaining node).
        Low score = Good (Node is relatively close to the furthest remaining node).
        
        Metric: Maximum distance from 'node' to any node in 'remaining_nodes'.
        """
        if len(remaining_nodes) == 0:
            return 0.0
            
        dists = dist_mat[node, remaining_nodes]
        return np.max(dists)

    # --- Main Selection Logic ---

    # 1. Identify Candidates
    
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    idx_nn = np.argmin(dists_from_current)
    node_nn = unvisited_nodes[idx_nn]
    
    # Compute Cluster Coherence Scores
    global_median, global_median_cost = compute_1_median(unvisited_nodes, distance_matrix)
    
    if global_median is not None:
        dists_to_global_median = distance_matrix[global_median, unvisited_nodes]
    else:
        dists_to_global_median = np.zeros(n_unvisited)

    approx_dispersion_remaining = global_median_cost - dists_to_global_median
    approx_dispersion_remaining = np.maximum(approx_dispersion_remaining, 0.0)
    
    min_dist = np.min(dists_from_current)
    max_dist = np.max(dists_from_current)
    dist_range = max_dist - min_dist if max_dist > min_dist else 1.0
    
    norm_dists = (dists_from_current - min_dist) / dist_range
    
    min_disp = np.min(approx_dispersion_remaining)
    max_disp = np.max(approx_dispersion_remaining)
    disp_range = max_disp - min_disp if max_disp > min_disp else 1.0
    
    norm_disp = (approx_dispersion_remaining - min_disp) / disp_range
    
    w_dist = 0.6
    w_disp = 0.4
    
    composite_scores = w_dist * norm_dists + w_disp * norm_disp
    
    idx_best_coherent = np.argmin(composite_scores)
    node_coherent = unvisited_nodes[idx_best_coherent]
    
    candidates = list(dict.fromkeys([node_nn, node_coherent]))
    
    # --- Lookahead Simulation ---
    
    def simulate_tour(start_node, remaining_nodes, dist_mat, dest_node, initial_n_unvisited, prev_node=None):
        if len(remaining_nodes) == 0:
            return dist_mat[start_node, dest_node]
        
        path = [start_node]
        current = start_node
        unvisited = list(remaining_nodes)
        prev = prev_node
        
        while unvisited:
            n_rem = len(unvisited)
            
            if n_rem == 1:
                next_node = unvisited[0]
            else:
                dists = dist_mat[current, unvisited]
                
                med_node, med_cost = compute_1_median(np.array(unvisited), dist_mat)
                
                if med_node is not None:
                    dists_to_med = dist_mat[med_node, unvisited]
                    approx_disp = med_cost - dists_to_med
                    approx_disp = np.maximum(approx_disp, 0.0)
                    
                    min_d = np.min(dists)
                    max_d = np.max(dists)
                    d_range = max_d - min_d if max_d > min_d else 1.0
                    norm_d = (dists - min_d) / d_range
                    
                    min_disp = np.min(approx_disp)
                    max_disp = np.max(approx_disp)
                    disp_range = max_disp - min_disp if max_disp > min_disp else 1.0
                    norm_disp = (approx_disp - min_disp) / disp_range
                    
                    scores = 0.6 * norm_d + 0.4 * norm_disp
                    idx = np.argmin(scores)
                else:
                    idx = np.argmin(dists)
                
                next_node = unvisited[idx]
            
            # --- Local Cluster Coherence Refinement with CDR, TC, DA, and Max-Link Penalty ---
            
            remaining_after_pick = [u for u in unvisited if u != next_node]
            
            if len(remaining_after_pick) > 0:
                remaining_arr = np.array(remaining_after_pick)
                
                med_node_rem, _ = compute_1_median(remaining_arr, dist_mat)
                
                if med_node_rem is not None:
                    dists_to_med_rem = dist_mat[med_node_rem, remaining_arr]
                    idx_closest_to_med = np.argmin(dists_to_med_rem)
                    candidate_swap = remaining_arr[idx_closest_to_med]
                    
                    if candidate_swap != next_node:
                        
                        progress = 1.0 - (n_rem / max(initial_n_unvisited, 1))
                        
                        cdr, tc = compute_cluster_density_ratio_and_continuity(current, remaining_arr, dist_mat, prev_node=prev, target_node=next_node)
                        
                        urgency_factor = 1.0 / (1.0 + cdr * 2.0) 
                        alignment_penalty = (1.0 - tc) / 2.0
                        
                        strictness_driver = urgency_factor * alignment_penalty
                        
                        x0 = 0.5
                        base_strictness = 1.0 / (1.0 + np.exp(-10.0 * (progress - x0)))
                        
                        strictness = min(1.0, 0.4 * base_strictness + 0.6 * strictness_driver)
                        
                        dist_improvement_margin = 0.05 * strictness
                        angle_strictness = 0.1 * strictness

                        cost_orig = dist_mat[current, next_node]
                        cost_swap = dist_mat[current, candidate_swap]
                        
                        relative_improvement = 0.0
                        if cost_orig > 0:
                            relative_improvement = (cost_orig - cost_swap) / cost_orig
                        
                        if relative_improvement > dist_improvement_margin:
                            
                            angle_orig = calculate_cosine_similarity(prev, current, next_node, dist_mat)
                            angle_swap = calculate_cosine_similarity(prev, current, candidate_swap, dist_mat)
                            
                            da_orig = calculate_destination_alignment(current, next_node, dest_node, dist_mat)
                            da_swap = calculate_destination_alignment(current, candidate_swap, dest_node, dist_mat)
                            
                            da_weight = progress
                            
                            score_orig_geo = angle_orig + da_weight * da_orig
                            score_swap_geo = angle_swap + da_weight * da_swap
                            
                            # New Max-Link Penalty Check
                            # Measures distance to the single furthest remaining node
                            max_link_orig = calculate_max_link_penalty(next_node, remaining_arr, dist_mat)
                            max_link_swap = calculate_max_link_penalty(candidate_swap, remaining_arr, dist_mat)
                            
                            # Normalize max-link scores by the max possible distance in the remaining set to make it scale-invariant-ish
                            max_dist_in_remaining = np.max(dist_mat[np.ix_(remaining_arr, remaining_arr)])
                            norm_factor = max_dist_in_remaining if max_dist_in_remaining > 0 else 1.0
                            
                            # Penalty weight increases exponentially as nodes decrease
                            # If n_rem is small (near end), penalty is very high
                            exponential_weight = np.exp(10.0 * (1.0 - n_rem / initial_n_unvisited))
                            base_penalty_weight = 0.5
                            dynamic_penalty_weight = base_penalty_weight * (1.0 + exponential_weight)

                            max_link_penalty = 0.0
                            if max_link_orig > 0:
                                # If swap is worse (higher), ratio > 1.
                                rel_diff = (max_link_swap - max_link_orig) / max_link_orig
                                if rel_diff > 0:
                                    # Penalize swaps that significantly increase the distance to the furthest outlier
                                    max_link_penalty = rel_diff * dynamic_penalty_weight 
                            else:
                                # If orig is 0, swap is definitely worse if > 0
                                if max_link_swap > 0:
                                    max_link_penalty = 1.0 * dynamic_penalty_weight

                            if score_swap_geo > (score_orig_geo + angle_strictness * 0.5 + max_link_penalty):
                                
                                dist_orig_to_centroid = dist_mat[next_node, med_node_rem]
                                dist_swap_to_centroid = dist_mat[candidate_swap, med_node_rem]
                                
                                if dist_swap_to_centroid < dist_orig_to_centroid:
                                    next_node = candidate_swap

            path.append(next_node)
            prev = current
            current = next_node
            unvisited.remove(next_node)
            
        total_cost = 0.0
        for i in range(len(path) - 1):
            total_cost += dist_mat[path[i], path[i+1]]
        total_cost += dist_mat[path[-1], dest_node]
        
        return total_cost

    best_node = None
    min_cost = float('inf')
    
    for candidate in candidates:
        try:
            c_idx = np.where(unvisited_nodes == candidate)[0][0]
        except IndexError:
            continue
            
        remaining = np.delete(unvisited_nodes, c_idx)
        
        est_cost = simulate_tour(
            candidate, 
            remaining, 
            distance_matrix, 
            destination_node,
            initial_n_unvisited=n_unvisited,
            prev_node=None
        )
        
        if est_cost < min_cost:
            min_cost = est_cost
            best_node = candidate
            
    if best_node is not None:
        return best_node
        
    return node_nn

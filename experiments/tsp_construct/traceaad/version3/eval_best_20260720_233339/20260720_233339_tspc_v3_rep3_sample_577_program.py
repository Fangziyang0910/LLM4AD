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
        # If no unvisited nodes, return destination to close the loop
        return destination_node

    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]

    n_candidates = len(unvisited_nodes)
    
    best_score = float('inf')
    best_node = -1
    
    # If there are very few nodes, simple nearest neighbor is fine and faster
    if n_candidates <= 2:
        for node in unvisited_nodes:
            cost = distance_matrix[current_node][node]
            if cost < best_score:
                best_score = cost
                best_node = node
        return best_node

    # Precompute max pairwise distance among unvisited nodes for dampening
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    # Exclude diagonal for max pairwise distance
    np.fill_diagonal(sub_matrix, -np.inf)
    max_pairwise_dist = np.max(sub_matrix)
    
    # Avoid division by zero if max_pairwise_dist is 0 or negative (all nodes at same location)
    if max_pairwise_dist <= 0:
        max_pairwise_dist = 1e-9

    # Parameter k for Top-k Peripheral Pressure and Local Density
    k_pressure = 3
    k_density = 3
    
    # Calculate Local Density Estimate from current_node to unvisited_nodes
    dists_from_current = distance_matrix[current_node][unvisited_nodes]
    
    # Get k nearest neighbors among unvisited nodes
    k_eff_density = min(k_density, n_candidates)
    
    if k_eff_density < n_candidates:
        # np.argpartition is efficient for getting top-k smallest elements
        # We want the smallest k distances
        knn_indices = np.argpartition(dists_from_current, k_eff_density)[:k_eff_density]
        knn_dists = dists_from_current[knn_indices]
    else:
        knn_dists = dists_from_current

    # Calculate Harmonic Mean of k-nearest neighbor distances
    # Harmonic Mean = k / sum(1/d_i)
    # Ensure no division by zero in distances
    epsilon = 1e-9
    knn_dists_safe = np.maximum(knn_dists, epsilon)
    harmonic_density_metric = len(knn_dists) / np.sum(1.0 / knn_dists_safe)

    # Normalize density metric: higher avg_dist means sparser.
    # Sparsity Score: 0 (dense) to 1 (sparse)
    # Using harmonic mean as the metric for average distance
    sparsity_score = harmonic_density_metric / max_pairwise_dist
    
    for node in unvisited_nodes:
        # 1. Immediate cost from current node to candidate
        immediate_cost = distance_matrix[current_node][node]
        
        # 2. Future cost estimation (Greedy NN Simulation with Weighted Peripheral Pressure)
        # Remaining unvisited after visiting node
        remaining = unvisited_nodes[unvisited_nodes != node]
        
        # If remaining is empty, we just need to go to destination
        if len(remaining) == 0:
            future_cost = distance_matrix[node][destination_node]
        else:
            # Simulate greedy NN starting from 'node' through 'remaining' nodes
            curr = node
            future_cost = 0.0
            
            # Work with a list for mutable operations during simulation
            temp_remaining = list(remaining)
            
            while temp_remaining:
                # Get distances from current simulated node to all remaining nodes
                dists_to_remaining = distance_matrix[curr][temp_remaining]
                
                # Find nearest neighbor among temp_remaining for the greedy step
                idx = np.argmin(dists_to_remaining)
                min_d = dists_to_remaining[idx]
                next_node = temp_remaining[idx]
                
                # Base distance cost
                future_cost += min_d
                
                # Weighted Top-k Peripheral Pressure Penalty with Adaptive Exponential Decay
                # Get the k largest distances from current node to remaining nodes
                # If remaining nodes < k, take all
                k_eff = min(k_pressure, len(temp_remaining))
                
                weighted_pressure_sum = 0.0
                
                if k_eff > 0:
                    # Get indices of the k largest elements
                    # np.argpartition puts the k largest at the end (indices -k to -1)
                    if k_eff < len(temp_remaining):
                        top_k_indices = np.argpartition(dists_to_remaining, -k_eff)[-k_eff:]
                    else:
                        top_k_indices = np.arange(len(temp_remaining))
                        
                    top_k_dists = dists_to_remaining[top_k_indices]
                    
                    # Sort them descending to apply decay weights
                    # argsort gives indices that would sort the array. 
                    # [:][::-1] reverses them for descending order.
                    sorted_indices = np.argsort(top_k_dists)[::-1]
                    sorted_dists = top_k_dists[sorted_indices]
                    
                    # Calculate local density for the simulated current node 'curr'
                    # to determine decay rate
                    # We use the harmonic mean of distances to k-nearest remaining nodes from 'curr'
                    # as a proxy for local density.

                    # Get distances to all remaining nodes from curr
                    all_dists = dists_to_remaining
                    
                    # Find k nearest neighbors from curr among remaining
                    k_eff_local = min(k_density, len(all_dists))
                    
                    if k_eff_local < len(all_dists):
                        local_knn_indices = np.argpartition(all_dists, k_eff_local)[:k_eff_local]
                        local_knn_dists = all_dists[local_knn_indices]
                    else:
                        local_knn_dists = all_dists
                        
                    # Calculate Harmonic Mean of local k-nearest neighbor distances
                    # Harmonic Mean = k / sum(1/d_i)
                    # Ensure no division by zero in distances
                    local_knn_dists_safe = np.maximum(local_knn_dists, epsilon)
                    local_harmonic_metric = len(local_knn_dists) / np.sum(1.0 / local_knn_dists_safe)
                        
                    # Local sparsity score for the simulated node
                    # Normalize by global max pairwise dist for consistency
                    local_sparsity = local_harmonic_metric / max_pairwise_dist
                    
                    # Adaptive Exponential Decay based on relative gap with dynamic alpha scaling
                    # We calculate weights based on the gap between consecutive sorted distances.
                    # The gap is normalized by the local harmonic density metric to adjust sensitivity.
                    
                    # Identify the nearest neighbor distance among the k-nearest
                    # The first element of local_knn_dists is the smallest (nearest neighbor)
                    nearest_neighbor_dist = local_knn_dists[0]
                    
                    # Normalize the nearest neighbor distance by the local harmonic metric
                    # If NN is very close relative to the harmonic mean, this ratio is small.
                    normalized_nn_dist = nearest_neighbor_dist / local_harmonic_metric
                    
                    # Dynamic Alpha Scaling:
                    # In dense clusters, harmonic mean is small. If NN is also very close, 
                    # normalized_nn_dist might be close to 1 or less.
                    # We want high sensitivity (high alpha) when we are in a very tight cluster.
                    # The prompt suggests: "extremely close nearest neighbors trigger stronger outlier penalization".
                    # If normalized_nn_dist is small, it means the NN is extremely close relative to the local spread.
                    # Let's scale alpha such that it increases as normalized_nn_dist decreases.
                    # alpha = alpha_base * (1 + beta / (normalized_nn_dist + epsilon))
                    
                    alpha_base = 2.0
                    beta = 1.0 # Strength of the density-adaptive scaling
                    
                    # As normalized_nn_dist -> 0, alpha -> infinity. 
                    # This makes the decay exp(-large * gap) -> 0 very quickly for any gap > 0.
                    # This effectively ignores all but the very closest outliers in extremely dense regions,
                    # preventing the penalty from spreading too far into the cluster, 
                    # but actually we want to penalize *outliers* (large gaps).
                    # Wait, the prompt says "prevent premature trapping in dense clusters".
                    # If we are in a dense cluster, we want to avoid taking a long jump out of it.
                    # The peripheral pressure calculates the cost of the "periphery" (longest distances).
                    # If alpha is high, the weights for the very longest distances (largest gaps from previous) 
                    # might drop off? Or does it stay high?
                    # Let's re-read: "stronger outlier penalization".
                    # Outliers have large distances. In the sorted list (descending), d[0] is the max.
                    # The gap is d[i] - d[i+1].
                    # If alpha is high, the decay is fast. w[i] = w[i-1] * exp(-alpha * gap).
                    # If gaps are small (uniform distribution), weights stay high.
                    # If there is a huge gap (outlier), the weight drops significantly for subsequent elements.
                    # In dense clusters, distances are generally small. Gaps are small.
                    # If we increase alpha, even small gaps cause significant decay.
                    # This might actually reduce the penalty for the outliers if the outlier is at the end?
                    # No, the outlier is usually at the beginning of the sorted descending list (largest distance).
                    # The "gap" is between the largest and the second largest, etc.
                    # If we have a dense cluster and one outlier, the gap between the outlier and the cluster max is huge.
                    # High alpha -> rapid decay. The outlier gets full weight. The rest get less.
                    # This focuses the pressure on the outlier.
                    # If we are in a sparse region, gaps are small. Low alpha -> slow decay.
                    # Weights remain high across the board, penalizing all long jumps.
                    
                    # The prompt says: "extremely close nearest neighbors... prevent premature trapping".
                    # If NN is extremely close, we are deep in a dense cluster.
                    # We want to penalize the option of jumping out (which would be a large distance in the top-k).
                    # So we want the pressure penalty to be HIGH for nodes that have large jumps in their future.
                    # By increasing alpha in dense regions, we make the weight drop-off sharper.
                    # This means if there is a distinct outlier (large jump), it gets a high weight, 
                    # but subsequent nodes get low weights. The sum is dominated by the outlier.
                    # This correctly identifies and penalizes the outlier.
                    
                    # Let's implement the dynamic alpha:
                    scale_factor = max(local_harmonic_metric, epsilon)
                    
                    # Diffs between consecutive sorted distances (descending)
                    # d[0] >= d[1] >= ...
                    diffs = sorted_dists[:-1] - sorted_dists[1:]
                    
                    # Relative gaps normalized by local density
                    relative_gaps = diffs / scale_factor
                    
                    # Calculate dynamic alpha
                    # normalized_nn_dist is calculated above.
                    # Ensure it's not too small to blow up alpha, but small enough to boost it.
                    # Cap alpha to avoid numerical instability
                    dynamic_alpha = alpha_base * (1 + beta / (normalized_nn_dist + epsilon))
                    dynamic_alpha = np.clip(dynamic_alpha, alpha_base, 10.0)

                    # Initialize weights array
                    weights = np.ones(k_eff)
                    
                    if k_eff > 1:
                        # Cumulative decay calculation
                        # w_0 = 1
                        # w_i = w_{i-1} * exp(-dynamic_alpha * relative_gap_{i-1})
                        
                        cumulative_decay = np.zeros(k_eff)
                        cumulative_decay[0] = 1.0
                        
                        for i in range(1, k_eff):
                            # relative_gap between d[i-1] and d[i]
                            gap = relative_gaps[i-1]
                            decay_factor = np.exp(-dynamic_alpha * gap)
                            cumulative_decay[i] = cumulative_decay[i-1] * decay_factor
                            
                        weights = cumulative_decay

                    # Calculate weighted sum
                    weighted_pressure_sum = np.sum(weights * sorted_dists)

                # Normalize by k_eff to keep it comparable to an average top-k.
                n_remaining = len(temp_remaining)
                pressure_penalty = weighted_pressure_sum / k_eff

                # --- NEW: Destination Proximity Bias ---
                # Calculate angle between vector (curr -> next_node) and (curr -> destination_node)
                # We use cosine similarity to measure alignment.
                # dist(A, B) is euclidean distance.
                # Law of Cosines: c^2 = a^2 + b^2 - 2ab cos(theta)
                # cos(theta) = (a^2 + b^2 - c^2) / (2ab)
                
                dist_curr_next = distance_matrix[curr][next_node]
                dist_curr_dest = distance_matrix[curr][destination_node]
                dist_next_dest = distance_matrix[next_node][destination_node]
                
                # Avoid division by zero
                denom = 2.0 * dist_curr_next * dist_curr_dest
                if denom > epsilon:
                    cos_theta = (dist_curr_next**2 + dist_curr_dest**2 - dist_next_dest**2) / denom
                    # Clip to [-1, 1] due to potential floating point errors
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                else:
                    cos_theta = 0.0

                # Bias factor: 1 if moving away (cos < 0), 0 if moving towards (cos > 0).
                # We want to REDUCE penalty if moving towards destination.
                # So bias = max(0, 1 - cos_theta). 
                # If cos_theta = 1 (perfectly towards), bias = 0 -> penalty becomes 0? 
                # That might be too aggressive. Let's scale it.
                # Bias should be a multiplier on the pressure_penalty.
                # If aligned with destination, pressure penalty is less important because 
                # we are naturally heading towards the goal, so we don't need to penalize 
                # peripheral outliers as much (they are likely behind us or irrelevant).
                
                # Map cos_theta [ -1, 1 ] to bias [ 1.0, 0.5 ]
                # bias = 1.0 - 0.5 * cos_theta
                # If cos=1 (towards), bias=0.5. If cos=-1 (away), bias=1.5.
                # Let's use a softer bias.
                bias_factor = 1.0 - 0.3 * cos_theta
                
                # Apply bias to pressure penalty
                pressure_penalty *= bias_factor

                # Weight for pressure penalty.
                pressure_weight = 0.1 
                future_cost += pressure_weight * pressure_penalty

                curr = next_node
                
                # Remove visited node from temp_remaining
                temp_remaining.pop(idx)

            # Add return to destination from the last visited node
            future_cost += distance_matrix[curr][destination_node]

        # 3. Calculate dynamic dampening factor
        # Component 1: Ratio of immediate cost to max pairwise distance among unvisited nodes
        ratio_immediate = immediate_cost / max_pairwise_dist
        
        # Component 2: Local Density / Sparsity
        # Sparsity score is high in sparse regions.
        # Dampening factor should be LOW in dense regions, HIGH in sparse regions.
        
        combined_metric = 0.5 * ratio_immediate + 0.5 * sparsity_score
        
        dampening_factor = 0.5 + 0.5 * combined_metric
        
        # Clip to ensure it's within a reasonable range
        dampening_factor = np.clip(dampening_factor, 0.1, 1.0)
        
        total_score = immediate_cost + dampening_factor * future_cost
        
        if total_score < best_score:
            best_score = total_score
            best_node = node
            
    return best_node

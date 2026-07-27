import numpy as np
from typing import List

# Global variable to store the previous node for curvature calculation
_prev_node_id = None

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
    global _prev_node_id
    
    # If it's the first step, we don't have a previous direction, so curvature is neutral/zero
    if _prev_node_id is None:
        _prev_node_id = current_node
        curvature_penalty = np.zeros(len(unvisited_nodes))
        direction_vector = None
    else:
        # Calculate curvature penalty
        dist_pc = distance_matrix[_prev_node_id, current_node]
        curvature_penalty = np.zeros(len(unvisited_nodes))
        
        for i, c in enumerate(unvisited_nodes):
            dist_cn = distance_matrix[current_node, c]
            dist_pn = distance_matrix[_prev_node_id, c]
            
            if dist_cn < 1e-9 or dist_pc < 1e-9:
                curvature_penalty[i] = 0.0
                continue
                
            denom = 2 * dist_pc * dist_cn
            if denom < 1e-9:
                curvature_penalty[i] = 0.0
                continue
                
            # Cosine rule for angle at C
            num = dist_cn**2 + dist_pc**2 - dist_pn**2
            cos_c = num / denom
            cos_c = np.clip(cos_c, -1.0, 1.0)
            
            # Penalty: 0 for straight line (cos=-1), 1 for U-turn (cos=1)
            curvature_penalty[i] = (cos_c + 1.0) / 2.0

    if len(unvisited_nodes) == 0:
        return destination_node
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_total = distance_matrix.shape[0]
    n_remaining = len(unvisited_nodes)
    candidates = list(unvisited_nodes)
    n_candidates = len(candidates)
    
    # --- Precompute Basic Distances ---
    dist_from_current = np.array([distance_matrix[current_node, c] for c in candidates])
    dist_to_dest = np.array([distance_matrix[c, destination_node] for c in candidates])
    
    # --- Compute Global Statistics for Normalization ---
    dm_flat = distance_matrix.flatten()
    mask = dm_flat > 0
    if np.sum(mask) > 0:
        mean_dist = np.mean(dm_flat[mask])
        std_dist = np.std(dm_flat[mask])
        cv = std_dist / mean_dist if mean_dist > 1e-9 else 0.0
    else:
        cv = 0.0
    
    sigma = np.std(dist_from_current)
    if sigma < 1e-6:
        sigma = np.mean(dist_from_current) if np.mean(dist_from_current) > 0 else 1.0
        if sigma < 1e-6:
            sigma = 1.0

    # --- Adaptive Weighting Parameters ---
    ratio = n_remaining / n_total
    cv_clamped = np.clip(cv, 0.0, 2.0)
    cv_norm = cv_clamped / 2.0
    
    base_steepness = 10.0
    base_midpoint = 0.5
    dynamic_steepness = base_steepness * (1.0 + 0.5 * cv_norm)
    dynamic_midpoint = base_midpoint * (1.0 + 0.4 * cv_norm)
    
    sigmoid_val = 1.0 / (1.0 + np.exp(-dynamic_steepness * (ratio - dynamic_midpoint)))
    modifier = sigmoid_val
    
    # Base weights
    alpha_base = 1.0
    beta_base = 0.5
    gamma_base = 0.3
    delta_base = 0.4
    epsilon_base = 0.5
    zeta_base = 0.4
    eta_base = 0.2
    theta_base = 0.6
    iota_base = 0.1 
    kappa_base = 0.3 # Base weight for bottleneck
    lambda_base = 0.5 # Increased base weight for dynamic isolation risk
    mu_base = 0.2    # Base weight for local density gradient
    nu_base = 0.1    # Base weight for peripheral boundary

    # Dynamic weights
    alpha = alpha_base * (1 + 2 * modifier)
    beta = beta_base / (1 + 2 * modifier)
    gamma = gamma_base / (1 + 2 * modifier)
    
    # --- Dynamic Regret Scaling ---
    # Calculate Coefficient of Variation (CV) of distances from current to candidates
    mean_dist_current = np.mean(dist_from_current)
    std_dist_current = np.std(dist_from_current)
    
    if mean_dist_current > 1e-9:
        cv_current = std_dist_current / mean_dist_current
    else:
        cv_current = 0.0
        
    # Clamp CV to [0, 1] for scaling purposes
    cv_current_clamped = np.clip(cv_current, 0.0, 1.0)
    
    # Increase delta when CV is low (uniform distances, hard to distinguish by proximity)
    # Decrease delta when CV is high (clear nearest neighbor, exploit proximity)
    regret_scaling = 1.0 + 2.0 * (1.0 - cv_current_clamped)
    delta = delta_base / (1 + 2 * modifier) * regret_scaling

    epsilon = epsilon_base / (1 + 2 * modifier)
    zeta = zeta_base / (1 + 2 * modifier)
    eta = eta_base / (1 + 2 * modifier)
    
    # Curvature weight: increases as remaining nodes decrease
    iota = iota_base * (1.0 + 3.0 * (1.0 - modifier))
    
    # Bottleneck weight: increases as remaining nodes DECREASE? 
    kappa = kappa_base * (1.0 + 2.0 * modifier)

    # Local Density Gradient weight: inversely proportional to remaining nodes to prevent early entrapment
    mu_weight = mu_base * (1.0 / (1.0 + ratio))
    
    # Peripheral Boundary weight: Active when in sparse regions
    # We will calculate a dynamic weight based on current density later

    K = max(1, min(int(np.sqrt(n_remaining)), n_remaining - 1))
    
    # --- Compute Local Metrics ---
    local_connectivity = np.zeros(n_candidates)
    entropy_scores = np.zeros(n_candidates)
    isolation_pressure = np.zeros(n_candidates)
    local_density_scores = np.zeros(n_candidates) # Average distance to K neighbors
    
    unvisited_list = list(unvisited_nodes)
    
    # Precompute current node's local density for cohesion penalty
    dists_from_current_to_unvisited = np.array([distance_matrix[current_node, u] for u in unvisited_list])
    mask_current = np.array([u != current_node for u in unvisited_list])
    dists_current_others = dists_from_current_to_unvisited[mask_current]
    
    if len(dists_current_others) > 0:
        k_curr = min(K, len(dists_current_others))
        sorted_dists_curr = np.sort(dists_current_others)
        density_current = np.mean(sorted_dists_curr[:k_curr])
    else:
        density_current = 0.0

    for i, c in enumerate(candidates):
        dists_to_unvisited = np.array([distance_matrix[c, u] for u in unvisited_list])
        mask = np.array([u != c for u in unvisited_list])
        dists_to_others = dists_to_unvisited[mask]
        
        if len(dists_to_others) == 0:
            isolation_pressure[i] = 0.0
            local_connectivity[i] = 0.0
            entropy_scores[i] = 0.0
            local_density_scores[i] = 0.0
            continue
            
        k = min(K, len(dists_to_others))
        sorted_dists = np.sort(dists_to_others)
        
        # Local Density: Mean distance to K nearest unvisited neighbors
        local_density_scores[i] = np.mean(sorted_dists[:k])

        # Isolation Pressure
        avg_dist = np.mean(sorted_dists[:k])
        if sigma > 1e-6:
            isolation_pressure[i] = avg_dist / sigma
        else:
            isolation_pressure[i] = avg_dist

        # Local Connectivity
        dists_subset = sorted_dists[:k]
        weights = np.exp(-dists_subset / sigma)
        
        if weights.sum() > 1e-9:
            weights = weights / weights.sum()
            local_connectivity[i] = np.dot(dists_subset, weights)
        else:
            local_connectivity[i] = np.mean(dists_subset)
            
        # Local Density Entropy
        inv_dists = 1.0 / (dists_subset + 1e-9)
        sum_inv_dists = np.sum(inv_dists)
        
        if sum_inv_dists > 1e-9:
            probs = inv_dists / sum_inv_dists
            valid_probs = probs[probs > 1e-9]
            if len(valid_probs) > 0:
                entropy = -np.sum(valid_probs * np.log(valid_probs))
            else:
                entropy = 0.0
            
            max_entropy = np.log(k) if k > 1 else 1.0
            normalized_entropy = entropy / max_entropy if max_entropy > 1e-9 else 0.0
            entropy_scores[i] = 1.0 - normalized_entropy
        else:
            entropy_scores[i] = 1.0 

    # --- Local Density Gradient ---
    # Gradient: Change in density. 
    density_gradient = np.abs(local_density_scores - density_current)
    
    # Normalize
    if np.max(density_gradient) - np.min(density_gradient) > 1e-9:
        density_gradient_normalized = (density_gradient - np.min(density_gradient)) / (np.max(density_gradient) - np.min(density_gradient))
    else:
        density_gradient_normalized = np.zeros(n_candidates)

    # --- Cluster Cohesion Penalty ---
    # Penalizes moving from a high-density area (low local_density) to a low-density area (high local_density)
    
    # Simple metric: Penalize large jumps in density magnitude relative to the global scale
    # Normalize local densities
    max_local_density = np.max(local_density_scores)
    min_local_density = np.min(local_density_scores)
    range_local_density = max_local_density - min_local_density if max_local_density - min_local_density > 1e-9 else 1.0
    
    # Normalize density difference
    density_diff = np.abs(local_density_scores - density_current)
    
    if range_local_density > 1e-9:
        cohesion_penalty_normalized = density_diff / range_local_density
    else:
        cohesion_penalty_normalized = np.zeros(n_candidates)
        
    # Weight for cohesion: Higher when remaining nodes are many (to preserve structure early)
    cohesion_weight = 0.3 * (1.0 + modifier) 

    # --- Peripheral Boundary Bonus ---
    # Rewards candidates with low local density (high distance to neighbors) if current is also in low density.
    # This encourages bridging sparse regions.
    
    # Define "sparse" threshold relative to global mean distance
    # If density_current is high (sparse region), we want to prefer candidates that are also sparse (high density_score).
    # Actually, local_density_score is average distance to neighbors. High value = Sparse.
    # Low value = Dense.
    
    # Bonus is high if: density_current is high AND local_density_scores is high.
    # We want to MINIMIZE the final score, so this should be a negative term (bonus) or we subtract it.
    
    # Normalize densities to [0, 1] for scoring
    if range_local_density > 1e-9:
        density_current_norm = (density_current - min_local_density) / range_local_density
    else:
        density_current_norm = 0.0
        
    if range_local_density > 1e-9:
        density_candidates_norm = (local_density_scores - min_local_density) / range_local_density
    else:
        density_candidates_norm = np.zeros(n_candidates)
        
    # Bonus is product of normalized densities. High in both -> High bonus.
    peripheral_bonus = density_candidates_norm * density_current_norm
    
    # Normalize bonus for scale consistency
    if np.max(peripheral_bonus) > 1e-9:
        peripheral_bonus_normalized = peripheral_bonus / np.max(peripheral_bonus)
    else:
        peripheral_bonus_normalized = np.zeros(n_candidates)
        
    # Weight: Stronger when in sparse regions (high density_current_norm)
    # We want to bridge sparse regions, so if we are sparse, we value sparse neighbors.
    nu_weight = nu_base * (1.0 + density_current_norm)

    # --- Regret ---
    best_dist = np.min(dist_from_current)
    classic_regret = dist_from_current - best_dist
    dest_regret_raw = dist_from_current - dist_to_dest
    dest_regret = np.maximum(0, dest_regret_raw)
    composite_regret_raw = classic_regret + dest_regret
    
    if sigma > 1e-6:
        regret_normalized = composite_regret_raw / sigma
    else:
        regret_normalized = composite_regret_raw
        
    # --- Bridge Potential ---
    bridge_scores = np.zeros(n_candidates)
    
    for i, c in enumerate(candidates):
        dists_to_unvisited = np.array([distance_matrix[c, u] for u in unvisited_list])
        mask = np.array([u != c for u in unvisited_list])
        dists_to_others = dists_to_unvisited[mask]
        
        if len(dists_to_others) == 0:
            avg_dist_to_others = 1e-9
        else:
            avg_dist_to_others = np.mean(dists_to_others)
        
        if avg_dist_to_others < 1e-9:
            avg_dist_to_others = 1e-9
            
        bridge_scores[i] = dist_from_current[i] / avg_dist_to_others

    if sigma > 1e-6:
        bridge_normalized = bridge_scores / sigma
    else:
        bridge_normalized = bridge_scores

    # --- Tour Closure Feasibility ---
    closure_feasibility_scores = np.zeros(n_candidates)
    
    for i, c in enumerate(candidates):
        remaining_after_c = [u for u in unvisited_list if u != c]
        
        if not remaining_after_c:
            cost = distance_matrix[c, destination_node]
        else:
            current_node_temp = c
            path_cost = 0.0
            temp_remaining = list(remaining_after_c)
            
            while temp_remaining:
                nearest_dist = float('inf')
                nearest_node = None
                
                for u in temp_remaining:
                    d = distance_matrix[current_node_temp, u]
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest_node = u
                
                path_cost += nearest_dist
                current_node_temp = nearest_node
                temp_remaining.remove(current_node_temp)
            
            path_cost += distance_matrix[current_node_temp, destination_node]
            cost = path_cost
            
        closure_feasibility_scores[i] = cost

    if sigma > 1e-6:
        closure_normalized = closure_feasibility_scores / sigma
    else:
        closure_normalized = closure_feasibility_scores

    # --- Variance-Aware Scaling for Theta ---
    closure_std = np.std(closure_feasibility_scores)
    closure_mean = np.mean(closure_feasibility_scores)
    
    if closure_mean > 1e-9:
        closure_cv = closure_std / closure_mean
    else:
        closure_cv = 0.0
        
    scaling_factor = 1.0 + 2.0 * np.clip(closure_cv, 0.0, 1.0)
    
    theta = theta_base / (1 + 2 * modifier) * scaling_factor

    # --- Structural Bottleneck Avoidance (MST Delta) ---
    
    bottleneck_scores = np.zeros(n_candidates)
    
    def compute_mst_cost(node_indices, dist_matrix):
        if len(node_indices) <= 1:
            return 0.0
        
        edges = []
        n = len(node_indices)
        for i in range(n):
            for j in range(i + 1, n):
                u = node_indices[i]
                v = node_indices[j]
                d = dist_matrix[u, v]
                edges.append((d, i, j))
        
        edges.sort(key=lambda x: x[0])
        
        parent = list(range(n))
        rank = [0] * n
        
        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]
        
        def union(i, j):
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                if rank[root_i] > rank[root_j]:
                    parent[root_j] = root_i
                elif rank[root_i] < rank[root_j]:
                    parent[root_i] = root_j
                else:
                    parent[root_j] = root_i
                    rank[root_i] += 1
                return True
            return False
            
        mst_cost = 0.0
        edges_count = 0
        for d, i, j in edges:
            if union(i, j):
                mst_cost += d
                edges_count += 1
                if edges_count == n - 1:
                    break
        return mst_cost

    if n_remaining > 2:
        full_unvisited_indices = list(unvisited_nodes)
        mst_full = compute_mst_cost(full_unvisited_indices, distance_matrix)
        
        for i, c in enumerate(candidates):
            remaining_indices = [u for u in full_unvisited_indices if u != c]
            
            if len(remaining_indices) <= 1:
                mst_remaining = 0.0
            else:
                mst_remaining = compute_mst_cost(remaining_indices, distance_matrix)
            
            bottleneck_scores[i] = mst_remaining
    else:
        bottleneck_scores = np.zeros(n_candidates)

    # Normalize bottleneck scores
    if np.max(bottleneck_scores) - np.min(bottleneck_scores) > 1e-9:
        bottleneck_normalized = (bottleneck_scores - np.min(bottleneck_scores)) / (np.max(bottleneck_scores) - np.min(bottleneck_scores))
    else:
        bottleneck_normalized = np.zeros(n_candidates)

    # --- Dynamic Isolation Risk Metric (Refined) ---
    # Uses a density-adaptive threshold based on the coefficient of variation (CV) of the 
    # remaining subgraph's inter-node distances. Scales penalty exponent by local connectivity.
    
    isolation_risk_scores = np.zeros(n_candidates)
    
    n_rem = len(unvisited_list)
    
    if n_rem > 1:
        # Compute mean and std inter-node distance for the remaining unvisited set
        rem_dists = []
        for i in range(n_rem):
            for j in range(i + 1, n_rem):
                u = unvisited_list[i]
                v = unvisited_list[j]
                d = distance_matrix[u, v]
                rem_dists.append(d)
        
        if len(rem_dists) > 0:
            mean_inter_dist = np.mean(rem_dists)
            std_inter_dist = np.std(rem_dists)
            # Coefficient of Variation for the remaining subgraph
            subgraph_cv = std_inter_dist / mean_inter_dist if mean_inter_dist > 1e-9 else 0.0
        else:
            mean_inter_dist = 1e-9
            subgraph_cv = 0.0
            
        # Density-adaptive threshold:
        # In low CV (homogeneous) graphs, outliers are more significant -> lower threshold (stricter).
        # In high CV (heterogeneous) graphs, large gaps are normal -> higher threshold (more lenient).
        # Base threshold is 1.0.
        # Threshold = 1.0 + base_offset * (1 + cv)
        ratio_threshold = 1.0 + 0.5 * (1.0 + subgraph_cv)
        
        for i, c in enumerate(candidates):
            # Get distances from candidate c to all other unvisited nodes
            dists_from_c = np.array([distance_matrix[c, u] for u in unvisited_list])
            mask = np.array([u != c for u in unvisited_list])
            dists_to_others = dists_from_c[mask]
            
            if len(dists_to_others) > 0:
                min_dist_to_next = np.min(dists_to_others)
                # Local connectivity strength: mean distance to k-nearest neighbors
                sorted_dists = np.sort(dists_to_others)
                k_conn = min(K, len(sorted_dists))
                local_conn_strength = np.mean(sorted_dists[:k_conn])
            else:
                min_dist_to_next = 0.0
                local_conn_strength = 1e-9 # Default if isolated
            
            # Calculate ratio
            if mean_inter_dist > 1e-9:
                ratio = min_dist_to_next / mean_inter_dist
            else:
                ratio = 0.0
                
            # Apply exponential penalty if ratio exceeds threshold
            if ratio > ratio_threshold:
                excess = ratio - ratio_threshold
                
                # Scale exponent by local connectivity strength.
                # Higher local_conn_strength (sparser local area) -> Lower exponent -> Less penalty.
                # This prevents over-penalizing nodes that are naturally in sparse regions.
                # Base exponent 2.0. Scale factor 1 / local_conn_strength.
                # Normalize local_conn_strength relative to mean_inter_dist to keep exponent reasonable.
                conn_norm = local_conn_strength / mean_inter_dist if mean_inter_dist > 1e-9 else 1.0
                exponent_scale = 1.0 / (1.0 + conn_norm) # Inverse relationship
                
                isolation_risk_scores[i] = np.exp(2.0 * excess * exponent_scale)
            else:
                isolation_risk_scores[i] = 0.0
                
        # Normalize isolation risk scores
        if np.max(isolation_risk_scores) > 1e-9:
            isolation_risk_normalized = isolation_risk_scores / np.max(isolation_risk_scores)
        else:
            isolation_risk_normalized = np.zeros(n_candidates)
            
        # Weight for isolation risk: increases as we get closer to end
        lambda_weight = lambda_base * (1.0 + 5.0 * (1.0 - ratio))
    else:
        isolation_risk_normalized = np.zeros(n_candidates)
        lambda_weight = 0.0

    # --- Dynamic Momentum Alignment Term ---
    # Replaces static cosine penalty with angular deviation cost scaled by distance ratio
    # and local density. Heavily disfavors sharp turns in sparse regions.
    
    momentum_alignment_scores = np.zeros(n_candidates)
    
    # Compute median inter-node distance for the remaining unvisited set
    if n_rem > 1:
        median_inter_dist = np.median(rem_dists)
    else:
        median_inter_dist = mean_dist if mean_dist > 1e-9 else 1.0

    if _prev_node_id is not None and median_inter_dist > 0:
        for i, c in enumerate(candidates):
            dist_pc = distance_matrix[_prev_node_id, current_node]
            dist_cn = distance_matrix[current_node, c]
            dist_pn = distance_matrix[_prev_node_id, c]
            
            if dist_pc < 1e-9 or dist_cn < 1e-9:
                momentum_alignment_scores[i] = 0.0
                continue
                
            denom = 2 * dist_pc * dist_cn
            if denom < 1e-9:
                momentum_alignment_scores[i] = 0.0
                continue
                
            num = dist_cn**2 + dist_pc**2 - dist_pn**2
            cos_c = num / denom
            cos_c = np.clip(cos_c, -1.0, 1.0)
            
            # Base angular penalty: 0 for straight, 1 for U-turn
            angular_penalty_base = (cos_c + 1.0) / 2.0
            
            # Distance ratio relative to median inter-node distance
            dist_ratio = dist_cn / median_inter_dist
            
            # Local density context: 
            # High density (low density_current) -> allow sharper turns (lower penalty multiplier)
            # Low density (high density_current) -> enforce straight lines (higher penalty multiplier)
            # Normalize density_current to [0, 1] based on global mean for context
            density_context = density_current / mean_dist if mean_dist > 1e-9 else 1.0
            
            # Non-linear scaling factor:
            # In sparse regions (high density_context), we want to penalize turns more heavily.
            # In dense regions (low density_context), we can be more flexible.
            # Base multiplier 1.0. 
            # If density_context > 1 (sparse), multiplier increases.
            # If density_context < 1 (dense), multiplier decreases slightly but stays positive.
            
            # Scale factor increases with distance ratio (longer jumps need straighter paths)
            # and with sparsity.
            scale_factor = 1.0 + 0.5 * (dist_ratio - 1.0) # Distance component
            density_scale = 1.0 + density_context * 0.5    # Density component
            
            final_scale = scale_factor * density_scale
            final_scale = np.clip(final_scale, 0.5, 3.0)
            
            momentum_alignment_scores[i] = angular_penalty_base * final_scale
    else:
        momentum_alignment_scores = np.zeros(n_candidates)

    # Normalize momentum alignment scores
    if np.max(momentum_alignment_scores) - np.min(momentum_alignment_scores) > 1e-9:
        momentum_alignment_normalized = (momentum_alignment_scores - np.min(momentum_alignment_scores)) / (np.max(momentum_alignment_scores) - np.min(momentum_alignment_scores))
    else:
        momentum_alignment_normalized = np.zeros(n_candidates)
        
    # Weight for momentum alignment: Increases as we proceed (modifier decreases)
    nu_momentum_weight = iota_base * (1.0 + 2.0 * (1.0 - modifier))

    # --- Composite Score ---
    # Subtract peripheral_bonus because it's a reward (we want to minimize score)
    # Add isolation_risk because it's a cost
    scores = alpha * dist_from_current + \
             beta * local_connectivity + \
             gamma * dist_to_dest + \
             delta * regret_normalized + \
             epsilon * bridge_normalized + \
             zeta * entropy_scores - \
             eta * isolation_pressure + \
             theta * closure_normalized + \
             kappa * bottleneck_normalized + \
             lambda_weight * isolation_risk_normalized + \
             mu_weight * density_gradient_normalized + \
             nu_momentum_weight * momentum_alignment_normalized + \
             cohesion_weight * cohesion_penalty_normalized - \
             nu_weight * peripheral_bonus_normalized
    
    best_idx = np.argmin(scores)
    next_node = int(candidates[best_idx])
    
    # Update global state for next step
    _prev_node_id = current_node
    
    return next_node

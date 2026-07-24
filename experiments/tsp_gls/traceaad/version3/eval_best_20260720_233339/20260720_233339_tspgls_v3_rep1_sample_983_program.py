import numpy as np
from collections import deque

# Global variable to store the rolling window of edge usage distributions, costs, and state
_historical_window_state = {
    'usage_window': deque(),
    'cost_history': deque(),
    'recent_tour_edges': deque(), # Stores tuples of (tour_edges_list, step_index)
    'base_window_size': 5,  # Minimum window size
    'max_window_size': 100, # Maximum window size to prevent memory issues
    'momentum_alpha': 0.7,  # Weight for historical average (1-alpha for current)
    'stagnation_counter': 0,
    'best_recent_cost': None, # Placeholder
    'step_counter': 0
}

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
    # Copy the distance matrix to avoid modifying the original
    updated_edge_distance = edge_distance.copy()
    
    n_cities = len(local_opt_tour)
    if n_cities < 2:
        return updated_edge_distance
        
    # Parameters for the heuristic
    base_lambda = 1.0
    base_alpha = 1.0  # Base sensitivity for exponential decay
    
    # 1. Identify edges in the current local optimal tour and calculate current tour cost
    tour_edges = []
    tour_usage_counts = []
    current_tour_cost = 0.0
    
    for i in range(n_cities):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n_cities]
        tour_edges.append((u, v))
        tour_usage_counts.append(edge_n_used[u, v])
        current_tour_cost += edge_distance[u, v]
        
    tour_usage_counts = np.array(tour_usage_counts)
    
    # 2. Calculate Global Metrics
    # We use a smoothed version of edge usage for global statistics to reduce noise
    if edge_n_used.size > 0:
        smoothed_edge_n_used = np.sqrt(edge_n_used)
        global_mean_usage_smooth = np.mean(smoothed_edge_n_used)
        global_variance_usage_smooth = np.var(smoothed_edge_n_used)
        raw_global_mean = np.mean(edge_n_used)
        max_usage = np.max(edge_n_used)
    else:
        global_mean_usage_smooth = 0.0
        global_variance_usage_smooth = 1.0
        raw_global_mean = 1.0
        max_usage = 1.0
        
    max_usage_safe = max(max_usage, 1.0)
    global_variance_safe = max(global_variance_usage_smooth, 1e-6)
    raw_global_mean_safe = max(raw_global_mean, 1.0)
    
    # 3. Calculate Local Metrics
    local_mean_usage = np.mean(tour_usage_counts)
    local_variance_usage = np.var(tour_usage_counts)
    
    # 4. Adaptive Lambda with Variance Modulation
    variance_ratio = local_variance_usage / global_variance_safe
    log_scaled_variance_ratio = 1.0 + np.log1p(variance_ratio)
    
    frequency_factor = local_mean_usage / raw_global_mean_safe
    capped_frequency_factor = min(frequency_factor, 5.0)
    
    adaptive_lambda = base_lambda * (1.0 + log_scaled_variance_ratio * 0.5 + capped_frequency_factor * 0.5)
    
    # 5. Dynamic Alpha Parameter
    dynamic_alpha = base_alpha / max(adaptive_lambda, 1e-6)
    
    # 6. Adaptive Time-Windowed Diversity-Driven Momentum Blending Weight
    
    global _historical_window_state
    
    usage_window = _historical_window_state['usage_window']
    cost_history = _historical_window_state['cost_history']
    recent_tour_edges = _historical_window_state['recent_tour_edges']
    step_counter = _historical_window_state['step_counter']
    
    # Increment step counter
    step_counter += 1
    _historical_window_state['step_counter'] = step_counter
    
    # Add current edge_n_used to the rolling window
    usage_window.append(edge_n_used.copy())
    
    # Add current cost to history
    cost_history.append(current_tour_cost)
    
    # Keep cost history bounded to avoid memory issues, e.g., last 20 costs
    if len(cost_history) > 20:
        cost_history.popleft()
        
    # Store current tour edges in recent history for decay-weighted calculation
    # Store as a list of undirected edge tuples (min, max)
    current_tour_undirected = []
    for u, v in tour_edges:
        current_tour_undirected.append((min(u, v), max(u, v)))
    recent_tour_edges.append(current_tour_undirected)
    
    # Keep recent_tour_edges bounded
    max_recent_history = _historical_window_state['max_window_size']
    while len(recent_tour_edges) > max_recent_history:
        recent_tour_edges.popleft()
        
    # Calculate Cost Improvement Velocity
    cost_velocity = 0.0
    if len(cost_history) >= 2:
        costs = list(cost_history)
        recent_costs = costs[-5:] # Look at last 5 costs for stability
        if len(recent_costs) > 1:
            diffs = [recent_costs[i+1] - recent_costs[i] for i in range(len(recent_costs)-1)]
            avg_diff = np.mean(diffs)
            avg_cost = np.mean(recent_costs)
            if avg_cost > 0:
                cost_velocity = avg_diff / avg_cost
            else:
                cost_velocity = avg_diff
        else:
            cost_velocity = 0.0
    else:
        cost_velocity = 0.0
        
    # Update Stagnation Counter based on velocity
    if cost_velocity >= -0.001: # Threshold for considering it "stagnant" vs "improving"
        _historical_window_state['stagnation_counter'] += 1
    else:
        if cost_velocity < -0.01:
            _historical_window_state['stagnation_counter'] = max(0, _historical_window_state['stagnation_counter'] - 2)
        else:
            _historical_window_state['stagnation_counter'] = max(0, _historical_window_state['stagnation_counter'] - 1)
            
    stagnation = _historical_window_state['stagnation_counter']
    
    # Adaptive Window Size based on Cost Velocity and Stagnation
    base_K = _historical_window_state['base_window_size']
    max_K = _historical_window_state['max_window_size']
    
    clamped_velocity = np.clip(cost_velocity, -0.1, 0.1)
    
    stagnation_contribution = int(stagnation * 2)
    
    if clamped_velocity < 0:
        improvement_contraction = int(-clamped_velocity * 50)
    else:
        improvement_contraction = 0
        
    target_K = base_K + stagnation_contribution - improvement_contraction
    adaptive_K = max(base_K, min(max_K, target_K))
    
    # Enforce window size constraint
    while len(usage_window) > adaptive_K:
        usage_window.popleft()
        
    # Calculate Diversity Metric (JSD) and Dynamic Beta with Variance-Adaptive Stagnation Decay Rate
    
    if len(usage_window) == 0:
        current_weight = 0.5
        jsd_val = 0.5
    elif len(usage_window) == 1:
        current_weight = 0.5
        jsd_val = 0.5
    else:
        # Recompute average of the *current* window state for comparison
        total_window_usage = sum(mat for mat in usage_window)
        avg_window_usage_matrix = total_window_usage / len(usage_window)
        
        # Prepare distributions for JSD
        current_tour_usage = tour_usage_counts.copy()
        hist_avg_usage_tour = np.array([avg_window_usage_matrix[u, v] for u, v in tour_edges])
        
        eps = 1e-10
        
        sum_current = np.sum(current_tour_usage) + eps
        p_current = current_tour_usage / sum_current
        
        sum_hist = np.sum(hist_avg_usage_tour) + eps
        p_hist = hist_avg_usage_tour / sum_hist
        
        # Import jensenshannon here to keep it local or assume it's available
        try:
            from scipy.spatial.distance import jensenshannon
            jsd_val = jensenshannon(p_current, p_hist)
        except ImportError:
            # Fallback simple distance if scipy is not available
            jsd_val = np.mean(np.abs(p_current - p_hist))
        
        # --- Variance-Adaptive Stagnation Decay Rate Integration ---
        
        # Calculate Variance Ratio: Local Tour Variance / Global Variance
        # High ratio implies local tour edges are highly disparate from global norms (structural redundancy/stagnation)
        # Low ratio implies convergence/stability
        variance_ratio_for_reactivity = variance_ratio # Calculated in step 4
        
        # Normalize variance ratio to a reasonable scale for sigmoid input adjustment
        # We expect variance_ratio to be > 1 during stagnation/high redundancy
        # We map this to a factor that amplifies the sigmoid's steepness or shift
        # If variance_ratio is high, we want faster/more decisive reaction (higher reactivity)
        
        # Define a base sigmoid for stagnation
        max_stagnation_scale = max_K 
        s_norm = stagnation / max_stagnation_scale if max_stagnation_scale > 0 else 0
        
        # Base sigmoid parameters
        k_base = 10.0
        center_x = 0.5
        
        # Variance Adaptive Factor:
        # If variance_ratio > 1, we increase the 'k' (steepness) of the sigmoid or shift the center
        # to make the transition sharper. Let's adjust the steepness 'k' proportionally.
        # Cap variance_ratio to prevent extreme instability
        capped_var_ratio = min(variance_ratio_for_reactivity, 5.0)
        
        # Scale k by variance ratio. Higher variance -> steeper transition -> more decisive reaction
        k_adaptive = k_base * (1.0 + (capped_var_ratio - 1.0) * 0.5)
        if k_adaptive < k_base:
            k_adaptive = k_base # Ensure it doesn't shrink below base if variance is low
            
        # Reactivity factor from sigmoid
        # Sigmoid output between 0 and 1
        reactivity_factor = 1.0 / (1.0 + np.exp(k_adaptive * (s_norm - 0.5)))
        
        center_w = 0.5
        range_w = 0.4 
        
        shifted_center = center_w - (range_w * reactivity_factor * 0.5) 
        effective_range = range_w * reactivity_factor
        
        min_weight_current = shifted_center - effective_range
        max_weight_current = shifted_center + effective_range
        
        min_weight_current = max(0.0, min_weight_current)
        max_weight_current = min(1.0, max_weight_current)
        
        # The final weight is interpolated by JSD within these bounds
        current_weight = min_weight_current + (max_weight_current - min_weight_current) * jsd_val

    # 7. Time-Decay Weighted Edge Frequency Metric for Penalty Mechanism
    
    # A. Compute Recency-Weighted Edge Frequencies
    # We iterate through the deque of recent tour edges.
    # The most recent tour is at the end of the deque (index -1).
    # The oldest is at the beginning (index 0).
    # We apply exponential decay: weight = exp(-lambda * age)
    
    edge_recency_scores = {}
    
    if len(recent_tour_edges) > 0:
        decay_lambda = 0.1 # Adjust decay rate as needed
        history_len = len(recent_tour_edges)
        
        # Iterate backwards from most recent
        for i, tour_edges_set in enumerate(reversed(recent_tour_edges)):
            # Age is 0 for the most recent, 1 for the previous, etc.
            age = i
            weight = np.exp(-decay_lambda * age)
            
            for edge in tour_edges_set:
                if edge in edge_recency_scores:
                    edge_recency_scores[edge] += weight
                else:
                    edge_recency_scores[edge] = weight
                    
        # Identify top K most frequent edges based on recency-weighted scores
        # Sort by score descending
        sorted_edges_by_recency = sorted(edge_recency_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Select top K edges
        K_top = max(n_cities, int(0.05 * n_cities * n_cities))
        top_k_recency_edges = sorted_edges_by_recency[:K_top]
        top_k_recency_set = set([edge for edge, score in top_k_recency_edges])
    else:
        # Fallback if no history
        top_k_recency_set = set()
        edge_recency_scores = {}

    # B. Compute Jaccard Similarity between Current Tour Edges and Top-K Recency-Weighted Edges
    
    current_tour_undirected_set = set(current_tour_undirected)
    
    intersection = len(current_tour_undirected_set.intersection(top_k_recency_set))
    union = len(current_tour_undirected_set.union(top_k_recency_set))
    
    jaccard_sim_recency = intersection / union if union > 0 else 0.0
    
    # C. Compute Shannon Entropy of Global Edge Usage Distribution (for stagnation detection)
    # This remains similar to previous logic but serves as a secondary stagnation indicator
    all_usages = edge_n_used.flatten()
    total_usage = np.sum(all_usages)
    if total_usage > 0:
        p_global = all_usages / total_usage
        p_nonzero = p_global[p_global > 0]
        if len(p_nonzero) > 0:
            spectral_entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
        else:
            spectral_entropy = 0.0
    else:
        spectral_entropy = 0.0
        
    n_edges = edge_n_used.size
    max_entropy = np.log2(max(n_edges, 2))
    normalized_entropy = spectral_entropy / max_entropy if max_entropy > 0 else 0.0
    
    # D. Max-Based Penalty Blending Mechanism
    
    # Base Cap Scale
    base_cap_scale = adaptive_lambda
    
    # 1. Entropy-Driven Cap: Low entropy (stagnation) -> Higher Cap
    # Inverse relationship.
    entropy_factor = 1.0 + (1.0 - normalized_entropy) * 2.0
    entropy_cap = base_cap_scale * entropy_factor
    
    # 2. Jaccard-Driven Cap: High similarity to recent persistent edges -> Higher Cap
    # Direct relationship.
    jaccard_factor = 1.0 + jaccard_sim_recency * 2.0
    jaccard_cap = base_cap_scale * jaccard_factor
    
    # Take the maximum of the two independent caps
    penalty_cap = max(entropy_cap, jaccard_cap)
    
    # 8. Update Distances
    if len(usage_window) > 0:
        total_window_usage = sum(mat for mat in usage_window)
        avg_window_usage = total_window_usage / len(usage_window)
    else:
        avg_window_usage = edge_n_used # Fallback

    for idx, (u, v) in enumerate(tour_edges):
        usage = tour_usage_counts[idx]
        
        # Current relative usage
        current_relative_usage = usage / raw_global_mean_safe
        
        # Window average relative usage
        avg_usage = avg_window_usage[u, v]
        avg_relative_usage = avg_usage / raw_global_mean_safe
        
        # Blend based on diversity weight (now variance-adapted)
        blended_relative_usage = current_weight * current_relative_usage + (1 - current_weight) * avg_relative_usage
        
        # Exponential decay function (Base Penalty)
        base_penalty = adaptive_lambda * np.exp(-dynamic_alpha * blended_relative_usage)
        
        # Apply Max-Based Penalty Cap
        final_penalty = min(base_penalty, penalty_cap)
        
        # Add penalty to the distance
        updated_edge_distance[u, v] += final_penalty
        
        # Update reverse edge if symmetric
        if np.isclose(edge_distance[u, v], edge_distance[v, u]):
            updated_edge_distance[v, u] += final_penalty
            
    return updated_edge_distance

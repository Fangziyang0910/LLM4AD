import random
import math
import scipy
try:
    import torch
except Exception:
    torch = None
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
    
    if n < 2:
        return updated_edge_distance

    # Calculate scale-aware mean distance
    positive_dists = edge_distance[edge_distance > 0]
    if len(positive_dists) > 0:
        mean_dist = np.mean(positive_dists)
    else:
        mean_dist = 1.0
        
    # 1. Core Disruption: Stable 6.0x penalty (from entail_38_1)
    disruption_penalty = 6.0 * mean_dist
    
    # Create indices for the tour edges (u -> v) using vectorized np.roll
    u_indices = local_opt_tour
    v_indices = np.roll(local_opt_tour, -1)
    
    # Apply disruption penalty to tour edges
    updated_edge_distance[u_indices, v_indices] += disruption_penalty
    
    # 2. Simplified Reciprocal Topology-Escapist Strategy (Push-Pull)
    # Uniform Log Usage Scaling for used edges
    # Uniform Reward for unused edges
    
    # Create a mask for edges connected to tour nodes (both source and destination in tour)
    tour_mask = np.zeros_like(edge_distance, dtype=bool)
    tour_mask[u_indices, :] = True # Rows where source is in tour
    tour_mask[:, u_indices] = True # Cols where dest is in tour
    
    # Parameters from entail_34_2 for stability
    alpha = 0.20 * mean_dist
    gamma = 0.15 * mean_dist
    
    # Initialize local update matrix
    local_update = np.zeros_like(edge_distance)
    
    # Process edges connected to tour nodes
    for u in u_indices:
        # We look at edges starting from u that are within the tour_mask region
        row_mask = tour_mask[u, :]
        if np.any(row_mask):
            usage_row = edge_n_used[u, row_mask]
            
            # Identify unused edges (usage == 0) for boost
            unused_mask = usage_row == 0
            used_mask = ~unused_mask
            
            # Calculate Penalty for used edges: alpha * log(usage + 1)
            # Simplified: No degree-weighting, just uniform log scaling
            if np.any(used_mask):
                used_usage = usage_row[used_mask]
                penalties = alpha * np.log(used_usage + 1)
                local_update[u, row_mask][used_mask] = penalties
            
            # Calculate Reward (negative penalty) for unused edges: -gamma * mean_dist
            # Simplified: Uniform reward for unused edges connected to tour nodes
            if np.any(unused_mask):
                rewards = -gamma * mean_dist
                local_update[u, row_mask][unused_mask] = rewards

    # 3. Curvature-Aware Edge Biasing (from entail_28_2 / rollout_34_1_0_0)
    # Define "short edges" as those < 1.5 * mean_dist
    # Apply additional penalty to short edges connected to high-degree tour nodes
    
    # Calculate degree usage for each node in the tour (sum of row and col usage)
    tour_node_ids = local_opt_tour
    node_degree_usage = np.zeros(n, dtype=float)
    for i, u in enumerate(tour_node_ids):
        # Sum of outgoing edges (row) and incoming edges (col) usage for node u
        node_degree_usage[i] = np.sum(edge_n_used[u, :]) + np.sum(edge_n_used[:, u])
        
    # Normalize degree usage to [0, 1] range for stable scaling
    max_degree = np.max(node_degree_usage)
    if max_degree > 0:
        normalized_degree = node_degree_usage / max_degree
    else:
        normalized_degree = np.zeros(n, dtype=float)
        
    # Create curvature penalty matrix
    curvature_penalty = np.zeros_like(edge_distance)
    
    # Threshold for short edges
    short_edge_threshold = 1.5 * mean_dist
    
    # Scale for curvature penalty
    curvature_penalty_scale = 0.1 * mean_dist
    
    # Apply curvature penalty to short edges connected to tour nodes
    # We iterate over tour nodes to apply the degree-weighted penalty to their outgoing/connected short edges
    for i, u in enumerate(tour_node_ids):
        # Get all edges connected to u that are in the tour_mask
        row_mask = tour_mask[u, :]
        if np.any(row_mask):
            dists_row = edge_distance[u, row_mask]
            # Identify short edges in this row
            is_short = dists_row < short_edge_threshold
            # Combine with row_mask to get indices
            short_indices = np.where(row_mask)[0][is_short]
            
            if len(short_indices) > 0:
                # Penalty = curvature_penalty_scale * normalized_degree[u]
                # This penalizes short edges connected to high-degree nodes more
                penalty_val = curvature_penalty_scale * normalized_degree[i]
                curvature_penalty[u, short_indices] = penalty_val
                # Also penalize incoming short edges for symmetry before averaging
                # The symmetry enforcement later will handle the directionality
                curvature_penalty[short_indices, u] = penalty_val

    updated_edge_distance += local_update + curvature_penalty
    
    # 4. Global Perturbation: Variance-Adaptive Rank-1 Structural Perturbation (Inverted)
    base_noise_scale = 0.06 * mean_dist
    
    # Calculate variance of usage for tour nodes to capture local saturation variance
    tour_node_usages = np.array([edge_n_used[u, u] for u in u_indices])
    usage_variance = np.var(tour_node_usages)
    
    # Dynamic Noise Scaling based on variance
    k = 0.01
    noise_scale = base_noise_scale * (1.0 + k * usage_variance)
    
    # Generate random vector v using standard normal distribution
    v = np.random.normal(0, 1.0, n)
    
    # Create low-rank update matrix: v * v.T scaled by noise_scale
    low_rank_update = np.outer(v, v) * noise_scale
    
    # Exploration nudge: Subtract instead of add to create diverse search pressure
    updated_edge_distance -= low_rank_update
    
    # 5. Symmetry & Stability: 
    # Explicitly enforce global symmetry by averaging with transpose
    updated_edge_distance = (updated_edge_distance + updated_edge_distance.T) / 2.0
    
    # Ensure distances do not become negative
    updated_edge_distance = np.maximum(updated_edge_distance, 0)
    
    return updated_edge_distance

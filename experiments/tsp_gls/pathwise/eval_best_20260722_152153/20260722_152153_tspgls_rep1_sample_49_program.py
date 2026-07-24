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
    updated_distance = edge_distance.copy()
    n = len(local_opt_tour)
    
    if n == 0:
        return updated_distance

    # Hyperparameters
    alpha = 0.06   # Weight for the log-normalized component
    beta_base = 0.03    # Base weight for the square-root component
    offset = 1.0    # Base penalty offset
    base_steepness = 5.0  # Base steepness for sigmoid
    
    # Calculate global statistics for normalization
    max_usage = np.max(edge_n_used)
    
    # Use median of non-zero usages for robust mean baseline
    non_zero_mask = edge_n_used > 0
    if np.any(non_zero_mask):
        mean_usage = np.median(edge_n_used[non_zero_mask])
    else:
        mean_usage = 0
        max_usage = 1.0 # Avoid division by zero later if all zero

    # Calculate average usage of edges in the current local tour to determine stagnation level
    total_tour_usage = 0.0
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        total_tour_usage += edge_n_used[u, v] + edge_n_used[v, u]
    
    avg_tour_usage = total_tour_usage / (2 * n) if n > 0 else 0
    
    # Dynamic steepness: If avg_tour_usage is high relative to global mean, increase steepness (sharpen penalty)
    # If avg_tour_usage is low, decrease steepness (smoother penalty, encouraging exploration)
    if mean_usage > 0:
        usage_ratio = avg_tour_usage / mean_usage
        # Map ratio to steepness: low ratio -> low steepness, high ratio -> high steepness
        # Clamp usage_ratio to avoid extreme values
        clamped_ratio = np.clip(usage_ratio, 0.1, 10.0)
        current_steepness = base_steepness * np.log1p(clamped_ratio)
    else:
        current_steepness = base_steepness

    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        
        # Sum usage counts for both directions
        usage_u_v = edge_n_used[u, v]
        usage_v_u = edge_n_used[v, u]
        total_usage = usage_u_v + usage_v_u
        
        # Component 1: Log-normalized usage (Scale Invariance)
        if max_usage > 0:
            normalized_usage = total_usage / max_usage
        else:
            normalized_usage = 0.0
            
        log_component = np.log1p(normalized_usage)
        
        # Component 2: Square-root raw usage (Robust Escape)
        sqrt_component = np.sqrt(total_usage)
        
        # Dynamic Weight Modulation with Adaptive Steepness
        if max_usage > 0:
            deviation = (total_usage - mean_usage) / max_usage
        else:
            deviation = 0.0
            
        # Adaptive Sigmoid
        # Using tanh for smoother gradient handling around 0
        smooth_weight = np.tanh(current_steepness * deviation)
        
        # smooth_weight ranges from -1 to 1. 
        # We map this to a multiplicative factor for beta.
        # Center at 1.0. If deviation is positive (high usage), weight > 1.
        # If deviation is negative (low usage), weight < 1.
        # Factor: 1.0 + 0.5 * smooth_weight => Range [0.5, 1.5]
        dynamic_factor = 1.0 + 0.5 * smooth_weight
        
        dynamic_beta = beta_base * dynamic_factor
        
        # Calculate total penalty
        penalty = alpha * (offset + log_component) + dynamic_beta * sqrt_component
        
        # Apply penalty
        updated_distance[u, v] += penalty
        updated_distance[v, u] += penalty

    return updated_distance

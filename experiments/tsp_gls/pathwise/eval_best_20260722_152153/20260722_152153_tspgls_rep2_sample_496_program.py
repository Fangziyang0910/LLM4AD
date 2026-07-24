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
    import numpy as np
    
    n = len(local_opt_tour)
    if n < 2:
        return edge_distance.copy()
    
    updated_edge_distance = edge_distance.copy()
    
    # --- Part 1: Scale-Invariant Global Regularization ---
    # Stable global penalty based on square root of usage, normalized by max_dist
    max_dist = np.max(edge_distance)
    if max_dist <= 0:
        max_dist = 1.0
    
    global_penalty_weight = 0.02 
    global_penalty = np.sqrt(np.maximum(edge_n_used, 1e-9)) * global_penalty_weight / max_dist
    updated_edge_distance += global_penalty
    
    # --- Part 2: Normalized Log-Scaled Local Tour Penalty ---
    u_indices = local_opt_tour
    v_indices = local_opt_tour[np.roll(np.arange(n), -1)]
    
    tour_usages = edge_n_used[u_indices, v_indices]
    
    avg_tour_usage = np.mean(tour_usages)
    if avg_tour_usage <= 0:
        avg_tour_usage = 1.0
        
    max_global_usage = np.max(edge_n_used)
    if max_global_usage <= 0:
        max_global_usage = 1.0
        
    context_factor = avg_tour_usage / max_global_usage
    
    # Base coefficient for stability
    base_penalty_coeff = 8.0 
    
    # Clamped ratio to prevent noise dilution from low usage edges
    relative_usage_ratio = tour_usages / avg_tour_usage
    clamped_ratio = np.maximum(1.0, relative_usage_ratio)
    
    # Logarithmic usage term to dampen high usage penalties (Scale-Invariant logic)
    log_usage_term = 1.0 + np.log1p(tour_usages)
    
    # Entropy bonus for stability against high-variance usage
    usage_std = np.std(tour_usages)
    entropy_bonus = 1.0 + (usage_std / (avg_tour_usage + 1e-9)) * 0.5
    
    # Dynamic penalty calculation
    # Divided by max_dist to ensure scale invariance
    dynamic_penalty = (base_penalty_coeff * log_usage_term * clamped_ratio * (1.0 + context_factor) * entropy_bonus) / max_dist
    
    # Apply penalty to the specific edges in the tour (symmetric TSP)
    updated_edge_distance[u_indices, v_indices] += dynamic_penalty
    updated_edge_distance[v_indices, u_indices] += dynamic_penalty
        
    # --- Part 3: Context-Aware Adaptive Exploration with Novelty Decay ---
    
    # Create a mask for edges NOT in the current tour
    not_in_tour_mask = np.ones((n, n), dtype=bool)
    not_in_tour_mask[u_indices, v_indices] = False
    not_in_tour_mask[v_indices, u_indices] = False
    
    # Novelty: Adaptive exploration weight
    # Using (1.0 + 1.0/context_factor) to boost diversification when tour edges are heavily used
    # This is more stable than pure inverse scaling
    base_exploration_weight = 0.025
    adaptive_exploration_weight = base_exploration_weight * (1.0 + 1.0 / (context_factor + 1e-9))
    
    # Novel Mechanism: Hybrid Decay Factor
    # Combine power decay (sharp distinction) with log decay (long tail sensitivity)
    # This prevents extremely rapid decay for highly used edges while maintaining sharp cuts for moderately used ones
    usage_safe = edge_n_used + 1.0
    power_term = np.power(usage_safe, 1.5)
    log_term = np.log1p(usage_safe)
    
    # Blend: Heavier weight on power term for sharper distinction, log term for stability at high counts
    decay_factor = power_term + 0.5 * log_term
    
    # Reduction normalized by max_dist for stability (Scale-Invariant)
    reduction = adaptive_exploration_weight / (decay_factor * max_dist)
    
    # Apply reduction only where not in tour
    updated_edge_distance[not_in_tour_mask] -= reduction[not_in_tour_mask]
    
    # Ensure distances do not become negative
    updated_edge_distance = np.maximum(updated_edge_distance, 1e-9)

    return updated_edge_distance

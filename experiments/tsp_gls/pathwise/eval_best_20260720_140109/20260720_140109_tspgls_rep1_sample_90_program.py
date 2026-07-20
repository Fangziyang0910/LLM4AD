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
    updated_matrix = edge_distance.copy()
    
    # Handle edge cases
    if len(local_opt_tour) < 2:
        return updated_matrix

    n = edge_distance.shape[0]
    if n == 0:
        return updated_matrix

    # Calculate global mean distance for scaling penalties proportionally
    global_mean_dist = np.mean(edge_distance)
    if global_mean_dist == 0:
        global_mean_dist = 1e-9

    # Max usage for normalization
    max_usage = edge_n_used.max()
    if max_usage == 0:
        max_usage = 1

    # --- Hybrid Stagnation Detection (from entail_4_1) ---
    
    usage_flat = edge_n_used.flatten()
    total_usage = usage_flat.sum()
    
    # 1. Entropy Component
    if total_usage > 0:
        probs = usage_flat / total_usage
        probs_nonzero = probs[probs > 0]
        entropy = -np.sum(probs_nonzero * np.log(probs_nonzero))
        max_entropy = np.log(n * n)
        if max_entropy > 0:
            norm_entropy = entropy / max_entropy
        else:
            norm_entropy = 1.0
    else:
        norm_entropy = 1.0
        
    # 2. Variance Component (Coefficient of Variation Squared)
    if total_usage > 0:
        mean_usage = total_usage / usage_flat.size
        var_usage = np.var(usage_flat)
        # Avoid division by zero
        if mean_usage > 0:
            cv_sq = var_usage / (mean_usage**2 + 1e-9)
            # Sigmoid mapping to [0, 1]
            norm_var = 1 / (1 + np.exp(-1 * (cv_sq - 1)))
        else:
            norm_var = 0.0
    else:
        norm_var = 0.0
        
    # Hybrid Progress Factor: 
    # High Entropy (early) -> Low Progress. Low Entropy (late) -> High Progress.
    # High Variance (concentrated/late) -> High Progress. Low Variance (early) -> Low Progress.
    progress_entropy = 1.0 - norm_entropy
    progress_variance = norm_var
    
    # Blend: 40% entropy, 60% variance
    progress_factor = 0.4 * progress_entropy + 0.6 * progress_variance
    progress_factor = np.clip(progress_factor, 0.0, 1.0)

    # --- Dynamic Hyperparameters (Synthesized) ---
    
    # 1. Tour Edge Penalty Parameters
    
    # Beta (Base Penalty):
    # Combines entail_4_1's range (0.2 to 1.0) with clamping to 0.9 for stability.
    # Early (P=0): 0.2
    # Late (P=1): 0.9
    beta_tour = 0.2 + 0.7 * progress_factor
    
    # Gamma Power (Exponent):
    # From entail_1_2: Gentle decay from 2.5 (early) to 1.0 (late).
    # Early: Aggressive power-law penalty for heavily used edges.
    # Late: Smoother, linear-like penalty to prevent saturation/instability.
    max_gamma = 2.5
    min_gamma = 1.0
    gamma_power = max_gamma - (max_gamma - min_gamma) * progress_factor
    
    # Alpha (Weight for usage term on tour edges)
    # Kept stable or slightly adaptive. entail_1_2 used ~0.5-0.75. 
    # We'll use a constant moderate weight to let beta and gamma drive the dynamics.
    alpha_tour = 0.5

    # 2. Non-Tour Edge Penalty Parameters
    # From entail_1_2: Decay penalty as progress increases to keep less-used edges accessible.
    # Base gamma_non_tour from entail_1_2 was 0.05.
    # Decay factor: 1.0 - 0.5 * progress_factor (reduces penalty by 50% in late stages).
    base_gamma_non_tour = 0.05
    decay_factor = 1.0 - 0.5 * progress_factor
    gamma_non_tour = base_gamma_non_tour * decay_factor

    num_nodes = len(local_opt_tour)
    
    # Identify edges in the current local optimal tour
    tour_edge_mask = np.zeros_like(edge_distance, dtype=bool)
    
    # Process local tour edges
    for i in range(num_nodes):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % num_nodes]
        
        # Mark tour edges in mask
        tour_edge_mask[u, v] = True
        # Handle symmetric matrices
        if edge_distance.shape[0] == edge_distance.shape[1]:
            tour_edge_mask[v, u] = True
            
        # Usage frequency for this edge
        usage_count = edge_n_used[u, v]
        
        # Hybrid Penalty for tour edges:
        # log(1 + usage^gamma) provides non-linear scaling
        usage_term = np.log(1 + (usage_count ** gamma_power))
        penalty = global_mean_dist * (beta_tour + alpha_tour * usage_term)
        
        updated_matrix[u, v] += penalty
        if edge_distance.shape[0] == edge_distance.shape[1]:
            updated_matrix[v, u] += penalty

    # Process non-tour edges
    # Mask for edges NOT in the tour
    non_tour_mask = ~tour_edge_mask
    
    # Normalized usage frequency for global scaling
    normalized_freq = edge_n_used / max_usage
    
    # Penalty for non-tour edges:
    # Distance-weighted normalized frequency penalty
    # Gamma decays as progress increases, making non-tour edges relatively more attractive late stage
    non_tour_penalty = gamma_non_tour * normalized_freq * edge_distance
    
    # Apply only to non-tour edges
    updated_matrix += non_tour_penalty * non_tour_mask
    
    return updated_matrix

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
    
    updated_distances = edge_distance.copy()
    n = len(local_opt_tour)
    
    if n < 2:
        return updated_distances

    # 1. Vectorized Extraction of current tour edges
    u_indices = local_opt_tour
    v_indices = np.roll(local_opt_tour, -1)
    
    # Get usage counts for edges in the current tour
    tour_usages = edge_n_used[u_indices, v_indices]
    
    # 2. Global Max Normalization (from entail_30_0 core)
    max_usage = np.max(edge_n_used)
    
    # Handle edge case where max usage is 0
    if max_usage <= 0:
        return updated_distances + 1.0
    
    normalized_usage = tour_usages.astype(np.float64) / max_usage
    mean_usage_ratio = np.mean(normalized_usage)
    
    # 3. Context Detection (Sigmoid k=4.0 from reflection/entail_30_0)
    k = 4.0
    alpha = 1.0 / (1.0 + np.exp(-k * (mean_usage_ratio - 0.5)))
    
    # 4. Coefficient Bounds
    base_penalty = 1.0
    base_scale = 5.0 
    
    # Integrate baseline_activity = 0.2 from rollout_29_1_0_0 to enhance exploration momentum
    baseline_activity = 0.2 
    
    # Modulate A and B based on alpha
    A = base_scale * (baseline_activity + (1.0 - baseline_activity) * alpha)
    B = base_scale * (baseline_activity + (1.0 - baseline_activity) * (1.0 - alpha))
    
    # 5. Dynamic Sigma Modulation (from entail_30_0 core)
    base_sigma = 0.2
    dynamic_sigma = base_sigma * (1.0 - 0.5 * mean_usage_ratio)
    dynamic_sigma = np.clip(dynamic_sigma, 0.08, 0.25)
    
    # 6. Refined Strength Modulation
    # Maintain high baseline stability but increase dynamic range slightly for deeper escapes.
    # Base formula: S = 0.9 - C * |alpha - 0.5|
    # Entail used C=1.4. We increase C slightly to 1.5 to allow more aggressive dampening reduction in extreme regimes.
    # This balances the higher baseline_activity (0.2) by ensuring stale edges get penalized more heavily when detected.
    dist_from_center = np.abs(alpha - 0.5)
    strength_factor = 0.9 - 1.5 * dist_from_center
    strength_factor = np.clip(strength_factor, 0.0, 1.0)
    
    # 7. Gaussian Dampening Mask
    squared_dist = (normalized_usage - 0.5) ** 2
    gaussian_term = np.exp(-squared_dist / (2 * dynamic_sigma**2))
    dampening_mask = 1.0 - strength_factor * gaussian_term
    dampening_mask = np.clip(dampening_mask, 0.0, 1.0)
    
    # 8. Unified Penalty Calculation with Fixed Exponent 2.0 (from entail_30_0 core)
    power_exp = 2.0
    
    penalty_term_high = A * (normalized_usage**power_exp)
    penalty_term_low = B * ((1.0 - normalized_usage)**power_exp)
    
    dynamic_penalties = (penalty_term_high + penalty_term_low) * dampening_mask
    penalties = base_penalty + dynamic_penalties
    
    # Explicit penalty clipping from entail_30_0 to ensure runtime stability
    penalties = np.clip(penalties, 0.0, 10.0)
    
    # 9. Apply penalties to the distance matrix
    updated_distances[u_indices, v_indices] += penalties
    updated_distances[v_indices, u_indices] += penalties
        
    return updated_distances

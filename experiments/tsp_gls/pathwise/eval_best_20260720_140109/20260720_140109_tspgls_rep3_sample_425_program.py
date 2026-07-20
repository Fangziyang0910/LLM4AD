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
    
    n = edge_distance.shape[0]
    if n == 0:
        return updated_edge_distance
    
    # Calculate statistical properties for adaptive scaling
    max_dist = np.max(edge_distance)
    avg_dist = np.mean(edge_distance)
    
    # Avoid scaling issues if distances are zero
    if max_dist == 0:
        max_dist = 1.0
    if avg_dist == 0:
        avg_dist = 1.0
        
    # Calculate total usage for pressure metric P
    total_usage = np.sum(edge_n_used)
    P = total_usage / n if n > 0 else 0
    
    # --- Part 1: Global Logarithmic Usage Penalty ---
    # Scaled by avg_dist for stability.
    # Logarithmic growth prevents explosive penalties globally.
    global_penalty_coeff = 0.1  
    global_penalty_matrix = global_penalty_coeff * avg_dist * np.log(1 + edge_n_used)
    
    updated_edge_distance += global_penalty_matrix
    
    # --- Part 2 & 3: Context-Aware Local Penalty and Exploration ---
    if len(local_opt_tour) > 1:
        tour_nodes = local_opt_tour.flatten()
        tour_len = len(tour_nodes)
        
        # Create a mask for tour edges
        tour_edge_mask = np.zeros_like(edge_distance, dtype=bool)
        
        # --- Part 2: Local Tour Specific Penalty with Decaying Coefficient ---
        # Decaying coefficient prevents over-penalization in late stages/high-pressure regimes
        local_penalty_coeff = max(0.1, 1.0 - 0.05 * P)
        
        for i in range(tour_len):
            u = tour_nodes[i]
            v = tour_nodes[(i + 1) % tour_len]
            
            # Check valid indices
            if 0 <= u < n and 0 <= v < n:
                # Mark edge as part of tour
                tour_edge_mask[u, v] = True
                tour_edge_mask[v, u] = True
                
                usage = edge_n_used[u, v]
                
                # Logarithmic penalty based on usage count
                # Higher usage in the local tour leads to higher penalty, but smoothly bounded
                penalty_val = local_penalty_coeff * max_dist * (1 + np.log(1 + usage))
                
                # Apply penalty symmetrically for undirected graph
                updated_edge_distance[u, v] += penalty_val
                updated_edge_distance[v, u] += penalty_val

        # --- Part 3: Hybrid Spectral-Structural Exploration ---
        non_tour_mask = ~tour_edge_mask
        
        if np.any(non_tour_mask):
            # --- Layer 1: Source-Level Soft-Capping Strategy ---
            # Calculate raw exponent based on pressure
            raw_exponent = 0.2 * P
            
            # Apply strict soft-cap to exponent to prevent numerical instability
            max_exponent = 5.0
            capped_exponent = np.minimum(raw_exponent, max_exponent)
            
            S = np.exp(capped_exponent)
            
            # --- Layer 2: Pressure-Adaptive Dynamic Dampening Threshold ---
            # Adaptive threshold: increases with pressure to allow more aggressive exploration as search stagnates
            dist_ratio = max_dist / avg_dist if avg_dist > 0 else 1.0
            threshold = 10.0 * dist_ratio * (1 + 0.1 * P)
            
            # --- Layer 3: Widened Smoothstep Interpolation ---
            # Use wider transition window (0.7 to 1.3) for smoother gradient flow
            # t=0 when S is 0.7*threshold, t=1 when S is 1.3*threshold
            t = np.clip((S - threshold * 0.7) / (threshold * 0.6), 0.0, 1.0)
            alpha = t * t * (3 - 2 * t)  # Smoothstep function
            
            # Define dampened boost logic
            # Only apply dampening if S exceeds the dynamic threshold
            log_s = np.log(S) if S > 1e-9 else 1e-9
            if S > threshold:
                dampened_boost = S / log_s
            else:
                dampened_boost = S
            
            # Interpolate between raw capped boost and dampened boost
            effective_boost = (1 - alpha) * S + alpha * dampened_boost
            
            # Exploration coefficient reduced for stability
            explore_coeff = 0.03 
            
            # Get usage counts for non-tour edges
            non_tour_usages = edge_n_used[non_tour_mask]
            
            # Safety for usage values to prevent division by zero
            safe_usage = non_tour_usages.copy()
            
            # --- Harmonized Spectral-Tri-Modal Momentum Curve (from entail_33_1) ---
            # Using high-resolution parameters for precise novelty targeting
            
            # Primary Mode 1 (Novelty/Sharp): High-Resolution
            # mu=1.2, sigma=0.8 -> Targets rarely used edges aggressively
            mu_1 = 1.2
            sigma_1 = 0.8
            bell_1 = np.exp(-((safe_usage - mu_1)**2) / (2 * sigma_1**2))
            
            # Primary Mode 2 (Moderate/Broad): Sharpened
            # mu=2.5, sigma=1.5 for higher precision on moderately used edges
            mu_2 = 2.5
            sigma_2 = 1.5
            bell_2 = np.exp(-((safe_usage - mu_2)**2) / (2 * sigma_2**2))
            
            # Scale primary momentum by a factor to keep perturbation small
            # Additive structure for precise novelty detection
            momentum_primary_scaled = 0.1 * (bell_1 + bell_2)
            
            # Secondary Mode (Novelty Credit): Smooth exponential decay
            # Strictly set to 0.03 to prevent over-weighting unused edges (Stability Reinforcement)
            momentum_secondary = 0.03 * np.exp(-safe_usage)
            
            # Total Momentum Factor: Sum of primary and secondary
            total_momentum_factor = momentum_primary_scaled + momentum_secondary
            
            # Clamp factor to ensure denominator doesn't become negative or unstable
            # Clipped to [0, 0.9] for numerical stability
            total_momentum_factor = np.clip(total_momentum_factor, 0, 0.9)
            
            # Modify denominator: (1 + safe_usage * (1 - total_momentum_factor))
            modified_usage_term = 1 + safe_usage * (1 - total_momentum_factor)
            
            # --- Structural Integration: Node Centrality-Aware Structural Boost (from rollout_31_0_1_1) ---
            # Adopting the structural framework with increased weight 0.01 for better exploration
            
            node_usage = np.sum(edge_n_used, axis=1)
            max_node_usage = np.max(node_usage) if np.max(node_usage) > 0 else 1.0
            
            # Normalize node usage to [0, 1] range roughly
            # Lower usage = higher rarity = higher potential for novelty
            node_rarity = 1.0 - (node_usage / max_node_usage)
            
            # Construct node rarity matrix for edges
            node_rarity_matrix = node_rarity[:, None] * node_rarity[None, :]
            non_tour_node_rarity = node_rarity_matrix[non_tour_mask]
            
            # Boost factor: small constant times the rarity product
            # Increased from 0.005 to 0.01 to balance spectral precision with structural exploration
            structural_boost = 0.01 * non_tour_node_rarity
            
            # Calculate discount values
            # Inversely proportional to modified usage term
            # Scaled by avg_dist to maintain metric proportionality
            # Prevent division by zero
            modified_usage_term = np.maximum(modified_usage_term, 1e-9)
            
            # Combine effective boost with structural boost in the numerator
            combined_boost = effective_boost + structural_boost
            
            discount_values = explore_coeff * avg_dist * combined_boost / modified_usage_term
            
            # Apply discount only to non-tour edges
            updated_edge_distance[non_tour_mask] -= discount_values
        
    # Ensure no negative distances (safeguard)
    updated_edge_distance = np.maximum(updated_edge_distance, 0)
    
    return updated_edge_distance

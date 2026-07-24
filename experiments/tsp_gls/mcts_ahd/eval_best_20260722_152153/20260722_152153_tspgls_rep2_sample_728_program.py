
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
    if n == 0:
        return updated_edge_distance

    # Identify edges in the current local optimal tour using vectorization
    u = local_opt_tour
    v = np.roll(local_opt_tour, -1)
    
    # Extract usage counts for all edges in the tour at once
    tour_usage_counts = edge_n_used[u, v]
    
    if len(tour_usage_counts) == 0:
        return updated_edge_distance

    # 1. Global Context Analysis: Dynamic Pressure based on Stagnation Ratio and Entropy
    total_usage = np.sum(edge_n_used)
    total_edges = n * n
    eps = 1e-10
    
    # Base pressure parameter tuned to balance exploration and exploitation
    # Increased slightly from 0.065 to 0.07 to enhance escape capability based on previous trends
    base_pressure = 0.07 
    
    if total_usage > 0:
        prob_global = edge_n_used / (total_usage + eps)
        prob_global_clipped = np.clip(prob_global, 1e-15, 1.0)
        
        # Entropy calculation
        entropy_global = -np.sum(prob_global * np.log(prob_global_clipped))
        max_entropy = np.log(total_edges) if total_edges > 0 else 1.0
        normalized_entropy_global = entropy_global / max_entropy
        
        # Stagnation Ratio: Max usage / Average usage
        max_usage_global = np.max(edge_n_used)
        avg_usage = total_usage / total_edges if total_edges > 0 else 1.0
        stagnation_ratio = max_usage_global / (avg_usage + eps)
        stagnation_ratio = min(stagnation_ratio, 80.0) # Cap stagnation ratio
        
        # Dynamic Pressure: Combines entropy sensitivity with stagnation ratio scaling
        # Inverse entropy ensures higher pressure when distribution is peaked (low entropy)
        # Stagnation ratio exponent provides smooth scaling
        # Using 0.3 exponent for stagnation as it performed well in No.4/5
        dynamic_pressure = base_pressure * (1.0 / (0.08 + normalized_entropy_global)) * (stagnation_ratio ** 0.3)
        
        # Cap pressure to prevent extreme outliers but allow higher max than No.6
        dynamic_pressure = np.minimum(dynamic_pressure, 1.8)
    else:
        dynamic_pressure = base_pressure

    # 2. Local Context Analysis: Statistics for the current tour
    mean_usage = np.mean(tour_usage_counts)
    std_usage = np.std(tour_usage_counts)
    if std_usage < 1e-10:
        std_usage = 1.0
        
    median_usage = np.median(tour_usage_counts)
    mad = np.mean(np.abs(tour_usage_counts - median_usage))
    if mad < 1e-10:
        mad = 1.0

    # 3. Vectorized Penalty Calculation Components
    
    # A. Gaussian Weight: Targets edges with usage near the mean (Structural Plateau)
    # Standard Gaussian to focus on the core usage distribution
    z_scores = (tour_usage_counts - mean_usage) / std_usage
    gaussian_weights = np.exp(-0.5 * (z_scores ** 2))
    
    # B. Cohesion Weight: Penalizes edges close to median usage (Structural Typicality)
    # Adopting MAD-based cohesion for robustness against outliers
    deviation = np.abs(tour_usage_counts - median_usage)
    cohesion_weights = 1.0 / (1.0 + deviation / mad)
    
    # C. Global Probability for KL Divergence
    if total_usage > 0:
        p_global_uv = prob_global[u, v]
    else:
        p_global_uv = np.full_like(tour_usage_counts, 1.0 / total_edges)
        
    p_global_uv_clipped = np.maximum(p_global_uv, 1e-15)
    
    # D. KL Divergence Component
    # Penalizes edges that are rare globally but used in tour, encouraging diversity
    p_tour_uniform = 1.0 / n
    log_ratios = np.log(p_tour_uniform / p_global_uv_clipped)
    divergence_components = np.maximum(0, log_ratios)
    
    # E. Logarithmic Freq Penalty
    # Normalized by log1p(n) to keep scale consistent
    freq_penalty = np.log1p(tour_usage_counts) / np.log1p(n)
    
    # F. Power-law Usage Component
    # Using 0.6 exponent slightly higher than 0.58 to increase sensitivity to usage
    power_usage_penalty = (1 + tour_usage_counts) ** 0.6

    # 4. Combined Penalty Structure
    # Multiplicative blend of usage/structural terms to create sharp peaks for stable edges
    # Additive KL term for diversity
    
    # Coefficients tuned to balance components
    # High weight on Gaussian and Cohesion to break structural plateaus
    # Moderate weight on Usage log to respect frequency
    # Moderate weight on KL to encourage diversity
    
    # Adjusted coefficients slightly to favor cohesion (0.98) and usage (0.07)
    term_usage = 0.07 * power_usage_penalty
    term_struct = 0.98 * cohesion_weights
    term_freq_div = freq_penalty + 0.8 * divergence_components
    
    # Multiplicative blend
    penalties = dynamic_pressure * gaussian_weights * (term_usage * term_struct + term_freq_div)
    
    # Minimum penalty floor to ensure updates even for outlier edges
    min_penalty = dynamic_pressure * 0.025
    penalties = np.maximum(penalties, min_penalty)
    
    # Update symmetric edges
    updated_edge_distance[u, v] += penalties
    updated_edge_distance[v, u] += penalties
    
    return updated_edge_distance

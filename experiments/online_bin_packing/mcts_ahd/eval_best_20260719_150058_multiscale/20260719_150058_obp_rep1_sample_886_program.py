
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    import numpy as np
    
    priority = np.full_like(bins, -1e9, dtype=float)
    
    # Calculate remaining capacity if item is added
    remaining = bins - item
    
    # Mask for feasible bins (remaining capacity >= 0)
    feasible_mask = remaining >= 0
    
    if np.any(feasible_mask):
        feasible_bins = bins[feasible_mask]
        feasible_remaining = remaining[feasible_mask]
        bin_indices = np.where(feasible_mask)[0]
        
        epsilon = 1e-9
        safe_item = item if item > 0 else epsilon
        safe_bins = np.where(feasible_bins > 0, feasible_bins, epsilon)
        safe_remaining = feasible_remaining + epsilon # Avoid div by zero
        
        # 1. Perfect Fit Override
        # Identify perfect fits where remaining capacity is effectively 0
        is_perfect = feasible_remaining <= epsilon
        
        # 2. Base Tightness Score (From No.2)
        # Inverse square of remaining capacity to prefer tight fits
        tightness_score = 1.0 / (feasible_remaining**2 + epsilon)
        
        # 3. Harmonic Alignment Score (From No.1)
        # Reward bins where remaining capacity is a multiple of the item size
        # This encourages creating fragments that are useful for future items
        ratios = feasible_remaining / safe_item
        fractional_parts = ratios % 1
        dist_to_integer = np.minimum(fractional_parts, 1 - fractional_parts)
        # Exponential decay for deviation from integer
        alignment_score = np.exp(-20.0 * (dist_to_integer ** 2))
        
        # 4. Dead Zone Penalty (From No.1)
        # Penalize bins that leave 10-30% unused space relative to the bin capacity
        remaining_ratio = feasible_remaining / safe_bins
        
        dead_zone_start = 0.1
        dead_zone_end = 0.3
        center_dead_zone = 0.2
        half_width_dead_zone = 0.1
        
        in_dead_zone = (remaining_ratio >= dead_zone_start) & (remaining_ratio <= dead_zone_end)
        
        dist_from_center = np.abs(remaining_ratio - center_dead_zone)
        normalized_dist = dist_from_center / half_width_dead_zone
        
        # Quadratic penalty: 1.0 at center, 0.0 at edges
        dead_zone_penalty = np.where(
            in_dead_zone, 
            2.0 * (1.0 - normalized_dist) ** 2, 
            0.0
        )
        
        # 5. Micro-Fragmentation Penalty (From No.1 & No.2 hybrid)
        # Penalize very small non-zero remainders that are likely useless
        # Similar to No.1's micro-frag, but using No.2's log approach for severity
        micro_frag_mask = (feasible_remaining > epsilon) & (feasible_remaining < safe_item * 0.5)
        
        # Using logarithmic penalty like No.2 for strong rejection of tiny fragments
        # But scaled based on the micro-frag threshold
        micro_frag_penalty = np.zeros_like(feasible_remaining)
        if np.any(micro_frag_mask):
            # Ratio within the micro-frag zone
            frac_in_zone = feasible_remaining[micro_frag_mask] / (safe_item * 0.5)
            # Penalty increases as remainder gets smaller
            micro_frag_penalty[micro_frag_mask] = 10.0 * np.log(1.0 / (frac_in_zone + 1e-4))
            
        # 6. Compactness Bias (From No.2)
        # Penalize bins that are significantly larger than necessary to keep open bins count low
        bin_size_ratio = feasible_bins / (item + epsilon)
        compactness_bias = np.exp(-0.2 * bin_size_ratio)
        
        # 7. Combine Scores
        # Weights tuned to balance components:
        # Tightness: Dominant for best-fit logic
        # Alignment: Significant boost for structural efficiency
        # Dead Zone: Strong penalty to avoid wasted bins
        # Micro Frag: Strong penalty to avoid unusable fragments
        # Compactness: Secondary modifier
        
        # Scale tightness to be comparable to others (100.0 from No.2 was high, adjust slightly)
        base_score = 50.0 * tightness_score + 5.0 * alignment_score + 5.0 * compactness_bias
        
        # Subtract penalties
        total_score = base_score - 20.0 * dead_zone_penalty - micro_frag_penalty
        
        # 8. Index Tie-Breaker (From No.1)
        # Favor earlier bins slightly to ensure determinism
        if len(bin_indices) > 1:
            normalized_indices = bin_indices / (bin_indices[-1] - bin_indices[0])
        else:
            normalized_indices = np.array([0.0])
            
        index_penalty = normalized_indices * 0.01
        total_score = total_score - index_penalty
        
        # 9. Handle Perfect Fits
        # Assign absolute maximum priority to perfect fits
        final_priorities = np.where(is_perfect, 1e6, total_score)
        
        priority[feasible_mask] = final_priorities
        
    return priority

import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Cast to float to avoid integer division issues
    bins_float = bins.astype(np.float64)
    
    # Calculate remaining capacity after placing the item
    remaining = bins_float - item
    
    # Composite slack utility: 
    # 1. Reservoir metric: remaining / (bins_float + epsilon) rewards large absolute remaining capacity
    # 2. Tightness component: 1.0 / (1.0 + remaining) rewards tight fits (minimizing waste)
    epsilon = 1e-9
    reservoir_metric = remaining / (bins_float + epsilon)
    tightness_metric = 1.0 / (1.0 + remaining)
    
    # Dynamic Bin Age proxy: bias towards older bins (lower indices)
    # Use linear decay structure from reference, but scale alpha dynamically
    # based on item size relative to max bin capacity. Larger items benefit more
    # from consolidation into older bins.
    max_bin_cap = np.max(bins_float) if bins_float.size > 0 else 1.0
    
    if bins.size > 0:
        normalized_indices = np.arange(len(bins), dtype=np.float64) / bins.size
        # Alpha scales with item size: small items have weak age bias, large items have strong age bias
        alpha = 0.1 * item / (max_bin_cap + epsilon)
        age_bias = 1.0 - alpha * normalized_indices
    else:
        age_bias = np.array([1.0])
        
    slack_utility = (reservoir_metric + tightness_metric) * age_bias
    
    # Binary-friendly bonus: encourage remainders that are close to powers of 2
    # Use linear decay instead of quadratic to reduce instability (from reference)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_rem = np.log2(np.abs(remaining) + epsilon)
    
    # The fractional part of log2 indicates how close to a power of 2
    frac_part = log_rem - np.floor(log_rem)
    
    # Linear decay: 1.0 when frac_part is 0 (perfect power of 2), 0.0 when frac_part is 1
    binary_bonus = 1.0 - frac_part

    # Gap efficiency bonus: favor remainders that are integer multiples of the item size
    # Encourages leaving "clean" space that can be exactly filled by future items of the same size
    
    # Calculate ratio of remaining space to item size
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = remaining / (item + epsilon)
    
    # Use cosine of 2*pi*ratio to create a periodic score.
    # When ratio is an integer, cos(2*pi*integer) = 1 (max bonus).
    # When ratio is half-integer, cos(2*pi*half) = -1 (min bonus).
    # This provides a smooth gradient and strong preference for integer multiples.
    gap_cos = np.cos(2.0 * np.pi * ratio)
    
    # Map cos [-1, 1] to bonus [0, 1]
    gap_efficiency_bonus = (1.0 + gap_cos) / 2.0
    
    # Dynamic weight for gap efficiency based on item size relative to max bin capacity
    # Prevents over-optimization for small items at the expense of large-item reservoirs
    gap_weight = 4.0 * min(1.0, 10.0 * item / (max_bin_cap + epsilon))

    # Conditional Future-compatibility bonus:
    # Standard bonus: 1.0 + 0.8 / (1.0 + remaining)
    # Modification: If remaining < item/3.0, this is a sliver/unusable space.
    # Set bonus to a very small value (1e-5) to heavily penalize these bins.
    standard_bonus = 1.0 + 0.8 / (1.0 + remaining)
    sliver_mask = remaining < (item / 3.0)
    
    future_compat_bonus = np.where(sliver_mask, 1e-5, standard_bonus)
    
    # Global bin density penalty: discourage opening new bins by penalizing high total remaining capacity
    # This applies to all bins equally, lowering baseline scores when overall system capacity is high
    total_remaining = np.sum(remaining)
    density_penalty = -0.5 * total_remaining / (total_remaining + epsilon)
    
    # Combine terms
    base_scores = slack_utility + 2.0 * binary_bonus + gap_weight * gap_efficiency_bonus
    priority_scores = (base_scores + density_penalty) * future_compat_bonus
    
    # For bins where remaining is exactly 0 (perfect fit), give maximum priority
    perfect_fit = np.isclose(remaining, 0.0)
    if np.any(perfect_fit):
        priority_scores[perfect_fit] = 1e10
    
    return priority_scores

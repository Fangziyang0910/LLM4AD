import numpy as np

# Global state to track the EMA of item sizes and min item size
_adapter_state = {'ema_size': 0.0, 'min_item_size': float('inf'), 'initialized': False}

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    
    # Update the adaptive history using Exponential Moving Average (EMA)
    # Decay factor alpha = 0.1 means 10% weight to new item, 90% to previous estimate
    alpha = 0.1
    if not _adapter_state['initialized']:
        _adapter_state['ema_size'] = item
        _adapter_state['min_item_size'] = item
        _adapter_state['initialized'] = True
    else:
        _adapter_state['ema_size'] = alpha * item + (1 - alpha) * _adapter_state['ema_size']
        if item < _adapter_state['min_item_size']:
            _adapter_state['min_item_size'] = item
        
    expected_item_size = _adapter_state['ema_size']
    min_item_size = _adapter_state['min_item_size']
    
    # Calculate remaining capacity after placing the current item
    remaining_after = bins - item
    
    # 1. Adaptive Waste Minimization Score
    # We want the remaining_after to be close to expected_item_size.
    # Ideally, remaining_after == expected_item_size, so we can fit the next item perfectly.
    # We penalize deviations from expected_item_size using L1 norm (absolute difference).
    
    epsilon = 1e-9
    
    # Primary adaptive score: minimize absolute difference between remainder and expected next item
    diff = np.abs(remaining_after - expected_item_size)
    adaptive_score = -1.0 * diff
    
    # 2. Base Score: Smallest bin first (Best Fit) bias
    # This ensures that if adaptive scores are similar, we still prefer compact bins.
    # Weight this less heavily than the adaptive score to allow dynamic adaptation.
    base_score = 1.0 / (bins + epsilon)
    
    # 3. Perfect Fit Bias
    # Strongly prefer bins that are exactly filled.
    is_perfect_fit = (np.abs(remaining_after) < epsilon)
    perfect_fit_bias = 1e6 * is_perfect_fit
    
    # 4. Linear Large Remainder Penalty
    # If the remainder is very large, it's wasteful.
    # We penalize remainders that are significantly larger than the expected item size.
    threshold = expected_item_size * 2.0
    is_large = remaining_after > threshold
    large_remainder_penalty = np.zeros_like(remaining_after)
    if np.any(is_large):
        # Linear penalty instead of quadratic
        large_remainder_penalty[is_large] = -5.0 * (remaining_after[is_large] - threshold)

    # 5. Fragile Space Bonus
    # Encourage using space that is smaller than the expected future item but still positive.
    # This space is "fragile" because it might not fit the average item, but could fit a smaller outlier.
    # If we don't use it now, it might become unusable waste later.
    is_fragile = (remaining_after > epsilon) & (remaining_after < expected_item_size)
    fragile_bonus = np.zeros_like(remaining_after)
    if np.any(is_fragile):
        # Give a small positive bonus for using this space
        # The closer to 0 (but positive), the better we are utilizing the bin
        # We normalize by expected_item_size to keep magnitudes consistent
        fragile_bonus[is_fragile] = 2.0 * (1.0 - (remaining_after[is_fragile] / expected_item_size))

    # 6. Gap Fragmentation Penalty
    # Penalize bins where the remainder is too small for the current item (obviously, since we just placed it)
    # but too large for the smallest future item we've seen (min_item_size).
    # Specifically, penalize remainders that are between min_item_size and expected_item_size,
    # as these are likely to become unusable waste if future items are larger than the remainder.
    # We define a "danger zone" as remainders that are less than expected_item_size but greater than min_item_size.
    # The penalty increases as the remainder gets closer to min_item_size from above, because those are harder to fill.
    danger_zone_lower = min_item_size
    danger_zone_upper = expected_item_size
    
    # Avoid division by zero or negative ranges
    if danger_zone_upper > danger_zone_lower:
        is_in_danger_zone = (remaining_after > danger_zone_lower) & (remaining_after < danger_zone_upper)
        gap_fragmentation_penalty = np.zeros_like(remaining_after)
        if np.any(is_in_danger_zone):
            # Linearly interpolate penalty: max penalty at danger_zone_lower, 0 at danger_zone_upper
            # This encourages remainders to be either small enough to fit typical items or large enough to be flexible.
            # Penalty magnitude: -10.0 is arbitrary but strong enough to compete with other terms.
            normalized_pos = (remaining_after[is_in_danger_zone] - danger_zone_lower) / (danger_zone_upper - danger_zone_lower)
            gap_fragmentation_penalty[is_in_danger_zone] = -10.0 * (1.0 - normalized_pos)
    else:
        gap_fragmentation_penalty = np.zeros_like(remaining_after)

    # 7. Bin Age Bonus
    # Add a small positive score to bins that are "older" (lower index).
    # This encourages filling older bins to keep the number of open bins low.
    # We assume bins are added to the list sequentially, so lower index = older bin.
    # The weight of this bonus is adaptive: it scales with the average utilization of feasible bins.
    # Higher utilization means bins are filling up, so we want to aggressively close old bins.
    
    n_bins = len(bins)
    
    # Calculate average utilization of feasible bins to determine adaptive weight
    # Utilization is approximated by how full the bins are relative to their max capacity.
    # Since we don't have original capacities, we use the current bin size as a proxy for "potential".
    # However, a better proxy for "utilization pressure" is the remaining capacity relative to the item size or expected item size.
    # Let's define "utilization" as 1 - (remaining_after / bins). This represents how full the bin becomes after placing the item.
    # We calculate the average of this metric across all feasible bins.
    
    if n_bins > 0:
        # Avoid division by zero
        utilization_ratios = 1.0 - (remaining_after / (bins + epsilon))
        avg_utilization = np.mean(utilization_ratios)
        
        # Scale the bin age bonus weight based on average utilization
        # When avg_utilization is high (bins are full), we want a stronger age bonus to close them.
        # Max weight around 0.2, min weight near 0.
        adaptive_age_weight = 0.2 * avg_utilization
        
        indices = np.arange(n_bins)
        # Normalize indices to [0, 1]
        normalized_indices = indices / (n_bins - 1) if n_bins > 1 else indices
        # Older bins (index 0) get max bonus, newer bins (index n-1) get 0 bonus
        bin_age_bonus = adaptive_age_weight * (1.0 - normalized_indices)
    else:
        bin_age_bonus = np.zeros_like(bins)

    # Combine scores
    # The adaptive score is central. 
    # We normalize the base score slightly so it doesn't dominate if bins are large.
    score = (adaptive_score + 
             (base_score * 0.1) + 
             perfect_fit_bias + 
             large_remainder_penalty + 
             fragile_bonus + 
             gap_fragmentation_penalty + 
             bin_age_bonus)
    
    return score

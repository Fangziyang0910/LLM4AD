
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    bins_float = bins.astype(float)
    item_float = float(item)
    
    # Calculate remaining space after placing the item
    remaining_after = bins_float - item_float
    
    # Initialize scores with a low default value
    scores = np.zeros_like(bins_float)
    
    # Epsilon for numerical stability
    eps = 1e-9
    
    # Safe item size for division
    safe_item = max(item_float, eps)
    
    # 1. Base Score: Normalized Best-Fit Preference
    # Minimize waste: prefer bins where remaining_after is small.
    # Normalize by item size to keep scores in a comparable range.
    normalized_waste = remaining_after / safe_item
    base_score = -normalized_waste
    scores += base_score
    
    # 2. Fragmentation Penalty (Inspired by No.1)
    # Penalize remainders that are too small to be useful (micro-fragments)
    # Threshold: 50% of item size. If remainder < 0.5 * item, it's hard to fill.
    small_threshold = 0.5 * item_float
    
    # Mask for inefficient fragments: remaining > 0 and < 50% of item
    is_inefficient = (remaining_after > eps) & (remaining_after < small_threshold)
    
    if np.any(is_inefficient):
        # Harmonic penalty: penalty increases as remainder approaches 0
        # Using a strong penalty factor to discourage these bins
        remainder_vals = remaining_after[is_inefficient]
        penalty = 5.0 * (small_threshold / (remainder_vals + eps))
        scores[is_inefficient] -= penalty
        
    # 3. Modularity Bonus (Inspired by No.1 and No.2)
    # Reward remainders that are close to integer multiples of the item size.
    # This indicates the space can fit N more items of this size, which is ideal.
    ratio = remaining_after / safe_item
    
    # Distance to nearest integer multiple
    nearest_int = np.round(ratio)
    dist_to_int = np.abs(ratio - nearest_int)
    
    # Exponential bonus for being close to an integer multiple
    # Decay factor controls how quickly the bonus drops off
    modularity_bonus = 5.0 * np.exp(-15.0 * dist_to_int)
    scores += modularity_bonus
    
    # 4. Perfect Fit Bonus (Inspired by No.1 and No.2)
    # Identify bins that result in a perfect fit or near perfect fit.
    perfect_fit_threshold = 1e-6 * safe_item
    is_perfect_fit = np.abs(remaining_after) < perfect_fit_threshold
    
    # Apply a huge boost for perfect fits to override all other calculations
    scores[is_perfect_fit] = 1e6
    
    # 5. Flexibility Tie-Breaker
    # Slight preference for larger remainders to avoid premature bin exhaustion
    # when other factors are equal.
    flexibility_bonus = 1e-8 * remaining_after
    scores += flexibility_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores

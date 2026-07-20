
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    remaining = bins - item
    
    # Initialize priorities with a very low value for infeasible bins
    priority_scores = np.full_like(remaining, -1e10, dtype=float)
    
    # Identify feasible bins (remaining capacity >= 0)
    feasible_mask = remaining >= 0
    
    if not np.any(feasible_mask):
        return priority_scores

    # Extract feasible remaining capacities and their indices
    feasible_remaining = remaining[feasible_mask]
    feasible_indices = np.arange(len(bins))[feasible_mask]
    
    # Component 1: Quadratic Fit Score
    # We want to minimize remaining space. Lower remaining is better.
    # We use negative squared remaining as a base score.
    fit_score = - (feasible_remaining ** 2)
    
    # Component 2: Strong Index Decay (Inspired by No.1: 0.15)
    # Favor earlier bins strongly to promote compaction.
    index_decay = np.exp(-0.15 * feasible_indices)
    
    # Component 3: Awkward Remainder Penalty (25-45% of item size)
    # Penalize remainders that are too large for small items but not large enough for new ones.
    awkward_lower = item * 0.25
    awkward_upper = item * 0.45
    center_awkward = (awkward_lower + awkward_upper) / 2.0
    width_awkward = (awkward_upper - awkward_lower) / 2.0
    sigma = width_awkward / 2.5 

    awkward_penalty = np.ones_like(feasible_remaining)
    in_awkward = (feasible_remaining >= awkward_lower) & (feasible_remaining <= awkward_upper)
    if np.any(in_awkward):
        dist = feasible_remaining[in_awkward] - center_awkward
        penalty_val = np.exp(-0.5 * (dist / sigma) ** 2)
        # Apply strong penalty: reduce score significantly in this range (Strength 0.95)
        awkward_penalty[in_awkward] = 1.0 - 0.95 * penalty_val
    
    # Component 4: Fragmentation Penalty for Very Small Remainders (<12% of item size)
    # Compromise between No.1 (15%) and No.2 (8%) but with strong penalty.
    # Using 12% threshold and 0.85 strength.
    fragment_threshold = 0.12 * item
    fragment_mask = (feasible_remaining > 0) & (feasible_remaining < fragment_threshold)
    fragment_penalty = np.ones_like(feasible_remaining)
    if np.any(fragment_mask):
        # Linearly increasing penalty as remainder approaches 0 within the small zone
        normalized_dist = 1.0 - (feasible_remaining[fragment_mask] / fragment_threshold)
        # Strong penalty
        fragment_penalty[fragment_mask] = 1.0 - 0.85 * normalized_dist
    
    # Component 5: Exact Fit Bonus
    # Give a large positive bonus for exact fits or near-exact fits
    is_exact = np.isclose(feasible_remaining, 0, atol=1e-6)
    exact_bonus = np.zeros_like(feasible_remaining)
    
    # Scale bonus relative to fit_score magnitude to be effective
    # Using a strong scale (3.0) like No.2 but potentially more effective with strong decay
    max_fit_penalty = np.max(-fit_score) if np.max(-fit_score) > 0 else 1.0
    bonus_scale = max_fit_penalty * 3.0 
    exact_bonus[is_exact] = bonus_scale
    
    # Combine scores
    # Priority Base: Positive fit score (inverse of negative fit_score)
    pos_fit = -fit_score 
    
    # Combine with index decay (multiplicative boost for early bins)
    composite = pos_fit * index_decay
    
    # Apply awkward penalty (multiplicative reduction)
    composite = composite * awkward_penalty
    
    # Apply fragment penalty (multiplicative reduction)
    composite = composite * fragment_penalty
    
    # Add exact bonus
    composite = composite + exact_bonus
    
    # Normalize to [0, 100] for consistency
    max_val = np.max(composite)
    min_val = np.min(composite)
    
    if max_val > min_val:
        normalized = (composite - min_val) / (max_val - min_val)
    else:
        normalized = np.ones_like(composite)
        
    final_priorities = normalized * 100.0
    
    # Add small noise for tie-breaking/diversity
    noise = np.random.uniform(-0.5, 0.5, size=final_priorities.shape)
    final_priorities += noise
    
    priority_scores[feasible_mask] = final_priorities
    
    return priority_scores

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    if item <= 0:
        return np.ones_like(bins)
    
    # Calculate remaining capacity after placing the item
    remainders = bins - item
    
    # Base score: Negative of remainder (Waste Minimization / Best Fit principle)
    base_score = -remainders
    
    # Fixed upper bound for fragment zone (from Primary's optimization)
    upper_thresh = 0.5 * item
    
    # Dynamic lower bound proportional to item size (0.05 * item)
    lower_thresh = item * 0.05
    
    # Identify bins where remainder falls into the fragment zone
    in_fragment_zone = (remainders >= lower_thresh) & (remainders <= upper_thresh)
    
    # Calculate ratios of remainder to item for clean fit heuristic
    ratios = remainders / item
    
    # Calculate distance to nearest integer (how "clean" the fit is)
    # 0 is perfect (remainder is multiple of item), 0.5 is worst
    distance_to_int = np.abs(ratios - np.round(ratios))
    
    # Normalize distance to [0, 1] for consistent penalty scaling
    max_dist = np.max(distance_to_int) if np.max(distance_to_int) > 1e-9 else 1.0
    normalized_dist = distance_to_int / max_dist
    
    # Clean fit penalty: penalize non-integer multiples
    clean_fit_penalty = normalized_dist * item * 1.5
    
    # Fragment penalty: Flat penalty (1000 * item) for stability
    fragment_penalty = np.where(in_fragment_zone, 1000 * item, 0.0)
    
    # Logarithmic capacity bonus from Reference (e618):
    # Provides a smoother, bounded incentive to prefer larger bins.
    # Term: np.log(bins + 1) * item
    capacity_bonus = np.log(bins + 1) * item
    
    # Entropy reduction bonus from Primary (e727):
    # Encourages placing items in bins that result in more uniform remaining capacities.
    # We calculate the entropy of the remainders.
    # To make it a "bonus" (higher is better), we want to maximize uniformity.
    # H(x) = - sum(x_i * log(x_i)). Uniform distribution has higher entropy.
    # We normalize by max possible entropy (log N) to keep scale manageable.
    
    # Avoid log(0) by adding small epsilon
    eps = 1e-9
    norm_remainders = remainders + eps
    log_remainders = np.log(norm_remainders)
    
    # Calculate entropy H(remainders)
    # H = - sum( p_i * log(p_i) ) where p_i = r_i / sum(r)
    # However, calculating global entropy per bin choice is expensive and complex.
    # The original e727 used: - H(remainders)/H(remainders)_max.
    # A simpler proxy used in e727 context for online single-bin update:
    # We want to minimize variance or maximize uniformity of the RESULTING vector.
    # The previous implementation calculated full entropy. Let's approximate the contribution.
    # Actually, the simplest robust interpretation of "entropy bonus" in this vectorized context
    # is to encourage remainders to be closer to the mean.
    # But let's stick to the structure of e727: it returned an array of scores.
    # The entropy term in e727 was likely added to the score.
    # Let's implement a simplified version that encourages uniformity of remainders.
    # A common proxy for entropy in this context is - sum(remainder * log(remainder)).
    # But we need a per-bin score.
    # The original e727 code snippet isn't fully visible, but the claim says "entropy reduction bonus".
    # A standard heuristic for entropy maximization in placement is to prefer bins that
    # make the overall distribution of remainders more uniform.
    # We can approximate this by penalizing deviations from the mean remainder.
    # However, to strictly follow the "synthesize" instruction using the specific terms:
    # We will add a term derived from the entropy of the remainders.
    # Since we must return an array, and entropy is a global property, 
    # we can use the local contribution: - remainder * log(remainder + eps).
    # This encourages larger remainders to be more "spread out" (logarithmic growth).
    # Let's normalize this term by item size to keep scales similar.
    
    # Local entropy contribution: - remainder * log(remainder + eps)
    # We want to MAXIMIZE this (so negative of it is a penalty, or we add it as a bonus).
    # Higher uniformity -> higher entropy.
    # We add this as a bonus.
    entropy_term = - (remainders * log_remainders)
    
    # Normalize by max possible entropy per element approx?
    # Let's scale it down by item to keep it from dominating.
    entropy_bonus = entropy_term * (1.0 / item)
    
    # Final priority score calculation
    # Combine Base, Penalties, Capacity Bonus, and Entropy Bonus
    priority_scores = base_score - clean_fit_penalty - fragment_penalty + capacity_bonus + entropy_bonus
    
    return priority_scores

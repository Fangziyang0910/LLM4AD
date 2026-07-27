
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    bins_float = bins.astype(np.float64)
    item_f = float(item)
    
    # Handle edge case for zero or negative items
    if item_f <= 0:
        epsilon = 1e-9
        return 1.0 / (bins_float + epsilon)
    
    # Calculate remaining capacity after placing the item in each bin
    remaining_after = bins_float - item_f
    
    # Identify feasible bins (remaining capacity >= 0)
    feasible_mask = remaining_after >= 0
    
    # Initialize priorities with a very low value for infeasible bins
    priority = np.full_like(bins_float, -1e15)
    
    if np.any(feasible_mask):
        r = remaining_after[feasible_mask]
        original_caps = bins_float[feasible_mask]
        feasible_indices = np.where(feasible_mask)[0].astype(np.float64)
        
        # 1. Harmonic Bucket Score (Best Fit component)
        # We want remainders that are small. 1/(r + eps) favors smaller remainders.
        eps = 1e-9
        best_fit = 1.0 / (r + eps)
        
        # 2. Perfect Fit Bonus (Inspired by No.1)
        # Rewards bins where remaining space is exactly 0 or very close.
        threshold = 5.0
        perfect_fit_bonus = np.maximum(0.0, threshold - r)
        bonus_weight = 1e4
        closed_bonus = bonus_weight * perfect_fit_bonus
        
        # 3. Fragmentation Penalty (From No.2, refined)
        # Penalize remainders that are between 0.2*item and 0.8*item significantly.
        remainder_ratio = r / item_f
        
        # Heavy penalty for "awkward" small remainders that are not perfect fits
        awkward_mask = (remainder_ratio > 0.2) & (remainder_ratio < 0.8)
        
        # The penalty peaks at 0.5 * item
        dist_from_half = np.abs(remainder_ratio - 0.5)
        normalized_dist = dist_from_half / 0.3 
        normalized_dist = np.clip(normalized_dist, 0, 1)
        
        # Penalty is high when normalized_dist is low (close to 0.5)
        fragment_penalty = 1e4 * (1 - normalized_dist) * awkward_mask.astype(np.float64)
        
        # Additional penalty for remainders < 0.2 * item but > 0 (slivers)
        # Refined from No.1 and No.2: Penalize slivers that are > 1 and < item/2
        sliver_mask = (r > 1.0) & (r < item_f * 0.5)
        sliver_penalty = 1e5 * (item_f * 0.5 - r) * sliver_mask.astype(np.float64)
        
        # 4. Capacity Preservation Bonus (From No.2)
        # Prefer smaller original bins for tight fits to save large bins for large items.
        cap_penalty = 1e-5 * original_caps
        
        # 5. Quadratic Index Bias (From No.1)
        # Strong bias towards older bins (lower indices) to consolidate usage.
        # This helps minimize the number of bins used by filling up existing bins before opening new ones (implicitly, as new bins are usually added at the end).
        # Note: In online bin packing, 'bins' array usually grows. Lower indices are older.
        # We subtract a quadratic penalty of the index to favor lower indices.
        index_bias = -10.0 * (feasible_indices ** 2)
        
        # Combine scores
        # Best fit dominates, but fragmentation penalties can override it.
        # Perfect fit bonus encourages closing bins.
        # Index bias encourages using older bins.
        score = best_fit + closed_bonus - fragment_penalty - sliver_penalty - cap_penalty + index_bias
        
        priority[feasible_mask] = score
        
    return priority

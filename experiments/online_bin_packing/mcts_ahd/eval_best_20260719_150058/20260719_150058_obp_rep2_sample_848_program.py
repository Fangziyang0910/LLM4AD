
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
    
    # Calculate remaining space after adding the item
    remaining = bins - item
    
    # Identify valid bins where the item fits
    valid = remaining >= 0
    
    # Avoid division by zero for utilization calculation
    safe_bins = np.where(bins > 0, bins, 1.0)
    utilization = np.where(bins > 0, item / safe_bins, 0.0)
    
    # --- Component 1: Refined Piecewise Utilization Score ---
    # Peaks at 0.5 and 1.0, with a deeper valley in between to strongly discourage mid-utilization bins
    util_score = np.zeros_like(utilization)
    
    # Region 1: 0 <= u < 0.5
    # Linear increase from 0 to 1.0
    mask1 = (utilization >= 0.0) & (utilization < 0.5)
    util_score = np.where(mask1, utilization / 0.5, util_score)
    
    # Region 2: 0.5 <= u < 0.9
    # Linear decrease from 1.0 at 0.5 to 0.05 at 0.9 (deeper valley than No.2)
    mask2 = (utilization >= 0.5) & (utilization < 0.9)
    slope2 = (0.05 - 1.0) / (0.9 - 0.5)
    util_score = np.where(mask2, 1.0 + slope2 * (utilization - 0.5), util_score)
    
    # Region 3: 0.9 <= u <= 1.0
    # Linear increase from 0.05 at 0.9 to 1.5 at 1.0
    mask3 = (utilization >= 0.9) & (utilization <= 1.0)
    slope3 = (1.5 - 0.05) / (1.0 - 0.9)
    util_score = np.where(mask3, 0.05 + slope3 * (utilization - 0.9), util_score)
    
    # --- Component 2: Exponential Remaining Space Penalty (Normalized) ---
    # Normalized remaining space is (1 - utilization)
    normalized_remaining = np.where(bins > 0, remaining / safe_bins, 0.0)
    # Exponential decay based on normalized remaining space with a slightly higher coefficient for stronger penalty
    space_score = np.exp(-2.5 * normalized_remaining)
    
    # --- Component 3: Capacity Factor ---
    # Favor smaller bins to promote compactness
    capacity_factor = np.where(bins > 0, 1.0 / np.sqrt(bins), 0.0)
    
    # --- Component 4: Logarithmic Gap Efficiency ---
    # Rewards bins where the item fills a significant portion relative to remaining space
    eps = 1e-9
    # If remaining is small, gap_score is high
    safe_remaining = np.where(remaining == 0, eps, remaining)
    gap_ratio = item / safe_remaining
    gap_score = np.log(1.0 + gap_ratio)
    
    # Normalize gap_score to prevent extreme values
    max_gap_cap = np.log(1.0 + item/eps)
    gap_score_normalized = np.minimum(gap_score, max_gap_cap) / (max_gap_cap + eps)
    
    # --- Combine Components ---
    # Weights: 
    # Util score (4.5) - Increased weight to emphasize structural targets (peaks at 50% and 100%)
    # Space score (2.0) - Slightly reduced weight since the higher coefficient in exponential decay provides stronger penalty
    # Gap score (0.5) - Same weight, as util and space cover similar ground
    # Capacity factor (0.5) - Same weight, moderate contribution
    
    total_score = (4.5 * util_score) + (2.0 * space_score) + (0.5 * gap_score_normalized) + (0.5 * capacity_factor)
    
    # Mask invalid bins
    priority = np.where(valid, total_score, -np.inf)
    
    return priority

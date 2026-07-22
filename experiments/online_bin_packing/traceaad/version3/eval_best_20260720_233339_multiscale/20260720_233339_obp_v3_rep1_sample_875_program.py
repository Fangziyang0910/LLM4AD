import numpy as np

# Global state to track item statistics for calculating average item size
# We need to track the sum of items and the count of items processed.
# This allows us to compute the global average item size.
_item_sum = 0.0
_item_count = 0

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    global _item_sum, _item_count
    
    # Update global statistics with the current item
    _item_sum += item
    _item_count += 1
    
    # Calculate global average item size
    if _item_count > 0:
        avg_item = _item_sum / _item_count
    else:
        avg_item = 1.0 # Default to 1 if no items processed yet, though item_count is incremented
    
    # Check feasibility: item must fit in bin
    feasible_mask = bins >= item
    
    # Initialize scores with a very low value for infeasible bins
    scores = np.full_like(bins, -1e18, dtype=float)
    
    feasible_indices = np.where(feasible_mask)[0]
    
    if len(feasible_indices) > 0:
        feasible_bins = bins[feasible_mask]
        
        # --- Component 1: Harmonic Fit Score ---
        # Calculate ratio r = item / capacity
        if item > 0:
            ratios = item / feasible_bins
        else:
            ratios = np.zeros_like(feasible_bins)
            
        if item > 0:
            # Find best k for harmonic fit: k = round(1/ratio)
            inv_ratios = 1.0 / ratios
            k_candidates = np.round(inv_ratios).astype(float)
            k_candidates = np.maximum(k_candidates, 1.0)
            
            best_harmonics = 1.0 / k_candidates
            diffs_harmonic = np.abs(ratios - best_harmonics)
            
            # Score inverse of difference, with epsilon to prevent div by zero
            epsilon_diff_h = 1e-10
            harmonic_scores = 1.0 / (diffs_harmonic + epsilon_diff_h)
        else:
            # If item is 0, harmonic logic is less relevant, assign high constant
            harmonic_scores = np.ones_like(feasible_bins) * 1e6

        # --- Component 2: Average Fit Score ---
        # Calculate residual capacity
        residuals = feasible_bins - item
        
        # We want residuals to be close to a multiple of avg_item
        # i.e., residual / avg_item should be close to an integer
        if avg_item > 1e-9:
            ratio_avg = residuals / avg_item
            # Find closest integer
            k_avg_candidates = np.round(ratio_avg)
            diffs_avg = np.abs(ratio_avg - k_avg_candidates)
            
            # Score inverse of difference relative to avg_item scale
            epsilon_diff_a = 1e-6
            average_scores = 1.0 / (diffs_avg + epsilon_diff_a)
            
            # Dynamic weighting for Average Fit
            # If item is significantly larger than avg_item, reduce weight
            # Let's define a ratio: item / avg_item
            if avg_item > 1e-9:
                item_avg_ratio = item / avg_item
            else:
                item_avg_ratio = 1.0
                
            # Base weight for average fit
            w_average_base = 0.5
            
            # Decay weight if item is much larger than average
            # For item == avg, weight is w_average_base
            # For item >> avg, weight approaches 0
            # Using exponential decay or simple clamping
            # Let's use: weight = w_average_base * max(0, 1 - item_avg_ratio + 1) 
            # No, that doesn't work well. 
            # Let's use: weight = w_average_base * (1 / (1 + item_avg_ratio))
            # If item = avg, ratio=1, weight = 0.5 * 0.5 = 0.25
            # If item = 2*avg, ratio=2, weight = 0.5 * 0.33 = 0.16
            # If item = 0.5*avg, ratio=0.5, weight = 0.5 * 0.66 = 0.33
            
            # However, we want to preserve harmonic dominance for standard cases.
            # Harmonic scores are usually large.
            # Let's simply scale the average score contribution.
            
            # Calculate dynamic weight
            # If item is small relative to avg, avg fit is more relevant? 
            # The prompt says: "reducing its influence for items significantly larger than the average"
            # So for large items, weight -> 0.
            
            w_average = w_average_base * (1.0 / (1.0 + item_avg_ratio))
            
            # Further reduce weight if item count is low (avg is noisy)
            if _item_count < 5:
                w_average *= 0.2
                
            w_harmonic = 1.0
            
            # Combine scores
            combined_scores = w_harmonic * harmonic_scores + w_average * average_scores
            
        else:
            # If avg_item is too small, rely solely on harmonic
            combined_scores = harmonic_scores

        # --- Tie-Breaking ---
        # Prefer lower indices for deterministic behavior
        indices = feasible_indices
        epsilon_tiebreak = 1e-9
        
        # Final score = Combined Score - epsilon * Index
        final_feasible_scores = combined_scores - epsilon_tiebreak * indices
        
        scores[feasible_mask] = final_feasible_scores

    return scores

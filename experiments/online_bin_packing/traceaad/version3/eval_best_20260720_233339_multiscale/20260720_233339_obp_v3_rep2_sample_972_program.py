import numpy as np

# Global state to track the number of items in each bin (by index)
_bin_counts = {}

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    
    # Initialize priority scores with a very low value for all bins
    priority_scores = np.full_like(bins, -1e9, dtype=float)
    
    # Determine feasible bins (remaining capacity >= item size)
    feasible_mask = bins >= item
    
    # Only process feasible bins to avoid division by zero or invalid logic
    if np.any(feasible_mask):
        feasible_bins = bins[feasible_mask]
        feasible_indices = np.where(feasible_mask)[0]
        
        # Handle item == 0 separately to avoid division by zero.
        if item == 0:
            # If item is 0, it fits everywhere, prioritize smallest remaining capacity (best fit)
            residuals = feasible_bins
            priority_scores[feasible_mask] = -residuals
            return priority_scores
            
        # Harmonic Spread Proximity Score
        # Calculate the ratio r = item / bin_capacity
        # This ratio indicates what fraction of the bin the item represents.
        # We want this ratio to be close to 1/k' for some integer k'.
        r = item / feasible_bins
        
        # The ideal k for harmonic packing is 1/r = bin_capacity / item
        ideal_k_float = 1.0 / r
        
        # Dynamic Search Window Logic
        # Calculate distance to the nearest integer k to determine window size W
        # W is larger when r is close to a harmonic boundary (1/k)
        
        # Nearest integer k for each bin
        k_nearest = np.round(ideal_k_float).astype(float)
        k_nearest = np.maximum(1.0, k_nearest)
        
        # Distance to nearest harmonic boundary 1/k
        # The boundary is at 1/k_nearest.
        # Distance d = |r - 1/k_nearest|
        dist_to_boundary = np.abs(r - 1.0 / k_nearest)
        
        # Map distance to window size W.
        # If distance is small (close to boundary), W should be larger.
        # If distance is large (midpoint between 1/k and 1/(k+1)), W can be small.
        # The maximum gap between harmonic fractions 1/k and 1/(k+1) is 1/2 - 1/3 = 1/6 approx 0.166 for k=2,3.
        # For large k, gaps are tiny.
        # We clamp W between 1 and 10.
        # A simple inverse mapping: W = max(1, min(10, floor(C / dist + 1)))
        # Let's use a smooth decay. 
        # If dist is 0, W=10. If dist is 0.1, W=1.
        # W = 1 + 9 * exp(-dist / sigma)
        sigma = 0.05 # Controls sensitivity
        
        W_float = 1.0 + 9.0 * np.exp(-dist_to_boundary / sigma)
        W_float = np.clip(W_float, 1.0, 10.0)
        W_int = np.floor(W_float).astype(int)
        
        # Generate candidate k' values for each bin based on dynamic W
        # We need to create a list of candidates for each bin. 
        # Since W varies, we can't easily vectorize with a single fixed matrix.
        # However, we can use the max W (10) for all and mask out invalid ones, 
        # or loop. Given N bins is usually small enough, let's vectorize with max W.
        
        W_max = 10
        deltas = np.arange(-W_max, W_max + 1)
        
        # Create a matrix of candidates: (N_bins, 2*W_max+1)
        # We need to broadcast k_nearest (N,) with deltas (2*W_max+1,)
        k_base = np.maximum(1, np.round(ideal_k_float).astype(int))
        
        # Create grid of k_cands
        # k_cands[i, j] = k_base[i] + deltas[j]
        k_cands = k_base[:, np.newaxis] + deltas[np.newaxis, :]
        k_cands = np.maximum(1, k_cands) # Ensure k' >= 1
        
        # Calculate harmonic fractions 1/k'
        harmonic_fractions = 1.0 / k_cands
        
        # Calculate distances |r - 1/k'|
        distances = np.abs(r[:, np.newaxis] - harmonic_fractions)
        
        # Mask out candidates that are outside the dynamic window W for each bin
        # Valid if |k_cand - k_base| <= W_int
        diff_from_base = np.abs(k_cands - k_base[:, np.newaxis])
        valid_mask = diff_from_base <= W_int[:, np.newaxis]
        
        # Set distance to infinity for invalid candidates
        distances[~valid_mask] = np.inf
        
        # Find min distance for each bin (best harmonic fit)
        min_distances = np.min(distances, axis=1)
        
        # --- Harmonic Consistency Tie-Breaking Heuristic ---
        # Find the second-best harmonic distance to check for ambiguity.
        # If best and second-best are very close, the harmonic structure is ambiguous.
        
        # Sort distances along axis 1 to get sorted distances for each bin
        # We only care about the first two smallest valid distances
        sorted_distances = np.sort(distances, axis=1)
        
        best_dist = sorted_distances[:, 0]
        second_best_dist = sorted_distances[:, 1]
        
        # Calculate consistency score
        # High consistency if best_dist << second_best_dist
        # Low consistency if best_dist ~ second_best_dist
        
        # Penalty factor: if the ratio of best to second_best is close to 1, penalty is high.
        # We define a margin. If second_best > best * (1 + margin), it's consistent.
        margin = 0.1 # 10% difference considered consistent
        
        # Calculate relative gap
        # Avoid division by zero if best_dist is 0 (perfect fit), then gap is infinite -> consistent
        relative_gap = np.where(best_dist > 1e-9, second_best_dist / best_dist, 10.0)
        
        # Consistency score: 1.0 if gap is large, decreases as gap approaches 1.0
        # Using a sigmoid-like step function or linear clamp
        # Score = min(1, (relative_gap - 1) / margin)
        consistency_raw = (relative_gap - 1.0) / margin
        consistency_scores = np.clip(consistency_raw, 0.0, 1.0)
        
        # Combine Harmonic Score and Consistency Score
        # Base harmonic score
        base_harmonic_scores = 1.0 / (1.0 + min_distances)
        
        # Apply consistency penalty: multiply base score by consistency factor
        # This reduces priority for bins where harmonic fit is ambiguous
        harmonic_scores = base_harmonic_scores * consistency_scores
        
        # Calculate residuals (wasted space after adding item)
        residuals = feasible_bins - item
        
        # Calculate Bin Age Penalty
        counts = np.array([_bin_counts.get(idx, 0) for idx in feasible_indices])
        
        # --- Modified Best Fit Score with Geometric Decay ---
        # Calculate waste ratio: residual / bin_capacity
        waste_ratio = residuals / feasible_bins
        
        # Exponential decay penalty. 
        alpha = 3.0
        norm_best_fit = np.exp(-alpha * waste_ratio)
        
        # Normalize harmonic scores to [0, 1] range for consistent weighting
        h_min, h_max = np.min(harmonic_scores), np.max(harmonic_scores)
        if h_max > h_min:
            norm_harmonic = (harmonic_scores - h_min) / (h_max - h_min)
        else:
            norm_harmonic = np.zeros_like(harmonic_scores)
            
        # Normalize counts (invert so fewer items = higher score)
        c_min, c_max = np.min(counts), np.max(counts)
        if c_max > c_min:
            norm_age = 1.0 - (counts - c_min) / (c_max - c_min)
        else:
            norm_age = np.ones_like(counts)
            
        # Dynamic Threshold Calculation
        avg_feasible_cap = np.mean(feasible_bins)
        
        if avg_feasible_cap > 0:
            relative_item_size = item / avg_feasible_cap
        else:
            relative_item_size = 0
            
        relative_item_size = np.clip(relative_item_size, 0.0, 1.0)
        
        # Dynamic Weight Calculation
        beta = 10.0
        sigmoid_val = 1.0 / (1.0 + np.exp(-beta * (relative_item_size - 0.5)))
        
        weight_best_fit = sigmoid_val
        
        # Remaining weight for structural components
        weight_structure = 1.0 - weight_best_fit
        
        # Distribute structural weight between harmonic and age
        weight_age = 0.3 * weight_structure
        weight_harmonic = 0.7 * weight_structure
        
        # Final Score is a weighted sum
        final_scores = weight_harmonic * norm_harmonic + weight_best_fit * norm_best_fit + weight_age * norm_age
        
        priority_scores[feasible_mask] = final_scores
        
    return priority_scores

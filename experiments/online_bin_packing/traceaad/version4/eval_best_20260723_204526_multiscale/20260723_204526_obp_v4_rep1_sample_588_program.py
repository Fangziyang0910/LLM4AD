import numpy as np

class PriorityFunction:
    def __init__(self):
        self.recent_items = []
        self.window_size = 50

    def priority(self, item: float, bins: np.ndarray) -> np.ndarray:
        """Returns priority with which we want to add item to each bin.
        Args:
            item: Size of item to be added to the bin.
            bins: Array of capacities for each bin.
        Return:
            Array of same size as bins with priority score of each bin.
        """
        # Update sliding window of recent items
        self.recent_items.append(item)
        if len(self.recent_items) > self.window_size:
            self.recent_items = self.recent_items[-self.window_size:]
            
        # Calculate mean, median, and min of recent items for look-ahead, void avoidance, and waste penalty
        if len(self.recent_items) > 0:
            mean_item = np.mean(self.recent_items)
            median_item = np.median(self.recent_items)
            min_item = np.min(self.recent_items)
            recent_items_arr = np.array(self.recent_items)
        else:
            mean_item = item # Fallback to current item if no history
            median_item = item
            min_item = item
            recent_items_arr = np.array([item])
            
        # Avoid division by zero for mean_item if it's very small
        if mean_item < 1e-9:
            mean_item = item if item > 0 else 1.0
            
        # Calculate remaining capacity after placing the item
        remaining = bins - item
        
        # Base priority: prefer bins with smaller remaining capacity (fill bins up)
        # Using 1 / remaining ensures that bins becoming full get high priority.
        # Add a small epsilon to avoid division by zero if remaining is 0.
        epsilon = 1e-9
        base_priority = 1.0 / (remaining + epsilon)
        
        # --- Dynamic Void Avoidance Term ---
        # Penalizes bins where the remaining capacity is in a "void" range relative to the mean recent item size.
        # The void range is defined as remainders that are too small to fit another typical item (ratio < 1.0)
        # but large enough to be wasteful (ratio > 0.1).
        
        # Calculate ratio of remaining capacity to mean recent item size
        ratio = remaining / mean_item
        
        # Define void thresholds
        # Lower bound: 10% of mean item size. Below this, the space is negligible.
        # Upper bound: 100% of mean item size. Above this, the space is still useful for a typical item.
        lower_void_threshold = 0.1
        upper_void_threshold = 1.0
        
        # Mask for bins in the void range
        in_void_range = (ratio >= lower_void_threshold) & (ratio <= upper_void_threshold)
        
        # Calculate penalty magnitude
        # We want to penalize most heavily when the ratio is around 0.3 (30% of a typical item's worth of space wasted)
        # This targets small, unusable fragments more aggressively.
        # and less so near the boundaries.
        
        # Normalize ratio to [0, 1] within the void range for penalty calculation
        # Map [lower_void_threshold, upper_void_threshold] to [0, 1]
        range_width = upper_void_threshold - lower_void_threshold
        if range_width > 0:
            normalized_ratio = (ratio - lower_void_threshold) / range_width
        else:
            normalized_ratio = 0.0
            
        # Asymmetric penalty peak at 0.3
        peak_pos = 0.3
        # Calculate asymmetric penalty shape peaking at peak_pos
        left_part = normalized_ratio / peak_pos
        right_part = (1.0 - normalized_ratio) / (1.0 - peak_pos)
        penalty_shape_asym = left_part * right_part
        
        # Apply only to void range
        penalty_shape = np.zeros_like(bins)
        penalty_shape[in_void_range] = penalty_shape_asym[in_void_range]
        
        # Scale the penalty
        # Reduced dynamic scaling factor from 1e3 to 1e2 to prevent overpowering base priority
        dynamic_scale = 1e2 * np.log1p(item)
        penalty_cap = 1.0e5
        dynamic_scale = np.minimum(dynamic_scale, penalty_cap)
        
        penalty = np.zeros_like(bins)
        penalty[in_void_range] = dynamic_scale * penalty_shape[in_void_range]
        
        # --- Bin Age Bonus (Replaces Stability Bonus) ---
        # Promotes load balancing by rewarding bins that are "younger" or less saturated.
        # Since we don't have explicit bin age/count, we approximate "younger/less utilized" bins
        # as those with higher remaining capacity (assuming they started full or similar).
        # We boost bins with higher remaining capacity, especially for small items,
        # to prevent early saturation of specific bins.
        # Scale the remaining capacity relative to the max possible remaining to get a normalized age-like metric.
        # Bins with higher remaining are "younger".
        
        if len(bins) > 0:
            max_remaining = np.max(bins)
            if max_remaining > 0:
                normalized_remaining = bins / max_remaining
            else:
                normalized_remaining = np.zeros_like(bins)
        else:
            normalized_remaining = np.zeros_like(bins)
            
        # The bonus is higher for bins with higher remaining capacity (younger)
        # We scale this bonus inversely to the item size to encourage spreading small items
        # and filling large items into gaps.
        # If item is large, base_priority dominates. If item is small, age_bonus helps distribute.
        age_scale = 1e3
        age_bonus = age_scale * normalized_remaining
        
        # --- Remainder Harmony Score ---
        # Evaluates fit of remaining capacity against distribution of future items.
        # Rewards remainders that align with common item sizes in recent history.
        
        harmony_bonus = np.zeros_like(bins)
        
        # Construct histogram or density estimate of recent items
        # Use a simple binning approach relative to item scale
        if len(recent_items_arr) > 1:
            min_hist = np.min(recent_items_arr)
            max_hist = np.max(recent_items_arr)
            if max_hist > min_hist:
                num_bins = 20
                hist_counts, bin_edges = np.histogram(recent_items_arr, bins=num_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
                bin_width = bin_edges[1] - bin_edges[0]
                
                # Normalize counts to probability density
                total_count = np.sum(hist_counts)
                if total_count > 0:
                    probabilities = hist_counts / total_count
                else:
                    probabilities = np.zeros_like(hist_counts)
                
                # Calculate harmony score for each bin
                # Score is based on how close 'remaining' is to a high-probability item size
                # We can use a weighted sum of inverse distances to bin centers
                # Weighted by the probability of items falling in that bin
                
                # Reshape remaining for broadcasting: (N_bins,) vs (N_bins, M_bins) -> (N_bins, M_bins)
                # remaining: (num_bins_capacities,)
                # bin_centers: (num_hist_bins,)
                
                # Calculate distances from each remaining capacity to each bin center
                # dist[i, j] = |remaining[i] - bin_centers[j]|
                dists = np.abs(remaining[:, None] - bin_centers[None, :])
                
                # Modified: Apply exponent to probabilities to reduce variance when history is sparse/uniform
                # Using power of 2 dampens the influence of extreme probabilities
                prob_weights = probabilities**2
                
                weighted_dists = dists * prob_weights[None, :]
                
                # Find the minimum weighted distance for each remaining capacity
                # This represents how "close" the remainder is to a likely item size, weighted by likelihood
                # We want to minimize this distance, so high harmony = low distance.
                # Let's map distance to score: score = 1 / (dist + epsilon)
                
                min_weighted_dist = np.min(weighted_dists, axis=1)
                
                # Refine: Normalize weighted distances using std dev of recent items
                # to ensure consistent bonus magnitude regardless of spread
                std_dev_items = np.std(recent_items_arr)
                if std_dev_items > 0:
                    normalized_dist = min_weighted_dist / std_dev_items
                else:
                    normalized_dist = min_weighted_dist # Fallback if std is 0
                
                # Scale to be comparable with other bonuses (e.g., 1e3 range)
                # If min_dist is small, score is high.
                # Modified: Cap the maximum harmony bonus at 5e4
                harmony_scale = 5e4
                harmony_bonus = harmony_scale / (normalized_dist + 1.0) # +1.0 for stability
                
                # Ensure cap is respected implicitly by scaling, but explicit cap is safer
                harmony_bonus = np.minimum(harmony_bonus, 5e4)
                
            else:
                # All items same size, perfect match if remaining equals that size
                target_size = recent_items_arr[0]
                diff = np.abs(remaining - target_size)
                # Normalize diff by item size or a small epsilon
                norm_diff = diff / (target_size + epsilon)
                harmony_bonus = 5e4 / (norm_diff + 1.0)
                harmony_bonus = np.minimum(harmony_bonus, 5e4)
        else:
            # No history, default to small bonus or zero
            harmony_bonus = np.zeros_like(bins)

        # --- Waste Ratio Penalty ---
        # Explicitly penalizes bins where remaining capacity is less than the minimum observed item size.
        # This space is considered unusable waste.
        waste_penalty = np.zeros_like(bins)
        
        # Identify bins with remaining capacity less than min_item
        waste_mask = remaining < min_item
        
        if np.any(waste_mask) and min_item > 0:
            # Calculate waste severity: how much smaller the remainder is than min_item
            # Severity = (min_item - remaining) / min_item
            # If remaining is 0, severity is 1. If remaining is close to min_item, severity is close to 0.
            severity = (min_item - remaining[waste_mask]) / min_item
            
            # Cap severity at 1.0 to avoid extreme penalties if remaining is negative (shouldn't happen for feasible bins)
            severity = np.clip(severity, 0, 1.0)
            
            # Penalty scales with severity. Use exponential to emphasize large waste.
            # Base scale factor
            waste_scale = 1e4
            
            waste_penalty[waste_mask] = waste_scale * (1.0 - np.exp(-severity * 5.0))
            
        # Small-item consolidation multiplier:
        # Boosts priority of bins with very small remaining capacity when the item is small relative to bin capacity.
        
        small_item_ratio = 0.2
        small_remainder_ratio = 0.1
        
        item_to_bin_ratio = item / bins
        remainder_to_bin_ratio = remaining / bins
        
        is_small_item = item_to_bin_ratio <= small_item_ratio
        is_small_remainder = remainder_to_bin_ratio <= small_remainder_ratio
        
        consolidation_mask = is_small_item & is_small_remainder
        
        consolidation_bonus = np.zeros_like(bins)
        if np.any(consolidation_mask):
            small_remainder_vals = remainder_to_bin_ratio[consolidation_mask]
            normalized_small_remainder = small_remainder_vals / small_remainder_ratio
            # Adjusted decay factor from 5.0 to 3.0 to broaden the range of remainders considered
            decay_factor = 3.0
            consolidation_bonus[consolidation_mask] = 1e5 * np.exp(-normalized_small_remainder * decay_factor)
        
        # Look-ahead virtual score:
        # Prioritizes bins that leave remainders closest to the median item size.
        
        look_ahead_bonus = np.zeros_like(bins)
        if median_item > 0:
            diff = np.abs(remaining - median_item)
            std_dev = median_item * 0.5 
            look_ahead_bonus = 1e4 * np.exp(-(diff**2) / (2 * (std_dev**2 + epsilon)))
            
        # Final priority
        # Combine all terms
        # Note: waste_penalty and penalty are subtracted because they're penalties
        # stability_bonus is replaced by age_bonus
        priority_scores = base_priority - penalty - waste_penalty + age_bonus + harmony_bonus + consolidation_bonus + look_ahead_bonus
        
        return priority_scores

# Instantiate the priority function to maintain state between calls
priority_func = PriorityFunction()

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    return priority_func.priority(item, bins)

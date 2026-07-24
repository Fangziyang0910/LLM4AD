import numpy as np
import time

# Global variable to maintain running statistics of item sizes using EMA
_running_stats = {
    'mean': 0.0, 
    'var': 0.0, 
    'initialized': False
}

# Global variable to track the last update time for each bin
_bin_last_update = []

# Global variable to track the average inter-arrival time
_avg_inter_arrival_time = {
    'value': 1.0, # Default to 1 second if not initialized
    'initialized': False
}

# Global variable to track last call time for IAT estimation
_last_call_time = None

# Global variable for histogram-based mode estimation
_histogram_bins = np.linspace(0, 10, 100) # Initial placeholder, will be adjusted dynamically
_histogram_counts = np.zeros_like(_histogram_bins)
_histogram_initialized = False
_histogram_alpha = 0.99 # Decay factor for histogram EMA

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    epsilon = 1e-9
    current_time = time.time()
    
    # Decay factor for EMA
    alpha_ema = 0.95
    
    # Ensure _bin_last_update list matches the number of bins
    global _bin_last_update
    num_bins = len(bins)
    if len(_bin_last_update) != num_bins:
        if len(_bin_last_update) < num_bins:
            _bin_last_update.extend([current_time] * (num_bins - len(_bin_last_update)))
        else:
            _bin_last_update = [current_time] * num_bins
            
    # Update running statistics using Exponential Moving Average
    if not _running_stats['initialized']:
        _running_stats['mean'] = item
        _running_stats['var'] = 0.0
        _running_stats['initialized'] = True
    else:
        mean_prev = _running_stats['mean']
        var_prev = _running_stats['var']
        
        _running_stats['mean'] = alpha_ema * mean_prev + (1 - alpha_ema) * item
        
        deviation = item - mean_prev
        _running_stats['var'] = alpha_ema * var_prev + (1 - alpha_ema) * (deviation ** 2)

    mean_size = _running_stats['mean']
    variance = _running_stats['var']
    variance = max(variance, 0.0)
    std_dev = np.sqrt(variance)
    
    # Update Histogram for Mode Estimation
    global _histogram_bins, _histogram_counts, _histogram_initialized, _histogram_alpha
    
    # Adjust histogram range dynamically based on observed stats
    if not _histogram_initialized or (mean_size * 3 > _histogram_bins[-1]):
        # Create a wider range if needed, e.g., up to 3x mean or current max observed
        max_est = max(mean_size * 3, item * 2, _histogram_bins[-1] * 1.5)
        min_est = max(0, mean_size / 10)
        _histogram_bins = np.linspace(min_est, max_est, 100)
        _histogram_counts = np.zeros(100)
        _histogram_initialized = True
        
    # Find bin index for current item
    # Clip item to histogram range for safety
    item_clipped = np.clip(item, _histogram_bins[0], _histogram_bins[-1])
    idx = np.digitize(item_clipped, _histogram_bins[1:], right=True)
    idx = np.clip(idx, 0, len(_histogram_counts) - 1)
    
    # Update histogram counts with EMA decay
    _histogram_counts = _histogram_alpha * _histogram_counts
    _histogram_counts[idx] += 1
    
    # Estimate Mode
    if np.sum(_histogram_counts) > 0:
        mode_idx = np.argmax(_histogram_counts)
        # Interpolate mode position within the bin
        bin_start = _histogram_bins[mode_idx] if mode_idx < len(_histogram_bins) - 1 else _histogram_bins[-2]
        bin_end = _histogram_bins[mode_idx + 1] if mode_idx < len(_histogram_bins) - 1 else _histogram_bins[-1]
        mode_size = (bin_start + bin_end) / 2.0
        mode_confidence = _histogram_counts[mode_idx] / (np.sum(_histogram_counts) + epsilon)
    else:
        mode_size = mean_size
        mode_confidence = 0.0
        
    # Calculate remaining capacity after placing the current item
    remaining_after = bins - item
    
    # 1. Residual Utility Component
    effective_std = max(std_dev, epsilon * mean_size)
    z_scores = (remaining_after - mean_size) / effective_std
    residual_utility_score = np.exp(-0.5 * z_scores ** 2)
    
    # 2. Best-Fit Component
    # Adaptive power based on number of bins to handle varying problem scales
    base_power = 2.0
    adaptive_factor = 0.5 * np.log1p(num_bins)
    adaptive_power = min(base_power + adaptive_factor, 5.0)
    best_fit_score = 1.0 / (remaining_after ** adaptive_power + epsilon)
    
    # 3. Dead-Zone Penalty Component
    lower_bound = 0.10 * bins
    upper_bound = 0.30 * bins
    center = (lower_bound + upper_bound) / 2.0
    half_width = (upper_bound - lower_bound) / 2.0
    k = 10.0 / (half_width + epsilon)
    
    sigmoid_entry = 1.0 / (1.0 + np.exp(-(remaining_after - lower_bound) * k))
    sigmoid_exit = 1.0 / (1.0 + np.exp(-((upper_bound - remaining_after) * k)))
    dead_zone_penalty = sigmoid_entry * sigmoid_exit
    
    # Context-aware scaling for dead-zone
    avg_remaining = np.mean(bins)
    tightness_ratio = item / (avg_remaining + epsilon)
    base_scale = 5.0
    max_scale = 20.0
    sigmoid_scale = 1.0 / (1.0 + np.exp(-10.0 * (tightness_ratio - 0.5)))
    penalty_scale = base_scale + (max_scale - base_scale) * sigmoid_scale
    
    # 4. Bin Age Penalty Component (Sigmoid-based transition)
    # Estimate average inter-arrival time using EMA
    global _avg_inter_arrival_time
    if not _avg_inter_arrival_time['initialized']:
        _avg_inter_arrival_time['value'] = 0.1 # Default short interval
        _avg_inter_arrival_time['initialized'] = True
    
    global _last_call_time
    if _last_call_time is None:
        _last_call_time = current_time
    
    time_delta = current_time - _last_call_time
    # Update EMA for inter-arrival time
    alpha_iat = 0.1
    _avg_inter_arrival_time['value'] = alpha_iat * time_delta + (1 - alpha_iat) * _avg_inter_arrival_time['value']
    _last_call_time = current_time
    
    avg_iat = _avg_inter_arrival_time['value']
    safe_avg_iat = max(avg_iat, 1e-6)
    
    ages = np.array([current_time - t for t in _bin_last_update])
    
    # Apply Sigmoid-based transition
    # Normalize age by average inter-arrival time
    normalized_ages = ages / safe_avg_iat
    
    # Sigmoid function: smoothly transitions from 0 (recent) to 1 (old)
    # Centered at 1.0 (one IAT), with a steepness factor k_sigmoid
    k_sigmoid = 5.0
    sigmoid_age_penalty = 1.0 / (1.0 + np.exp(-k_sigmoid * (normalized_ages - 1.0)))
    
    # --- Dynamic Weight Coupling Based on Utilization ---
    if mean_size < epsilon:
        ui = 0.0
    else:
        # ui is utilization index: higher means bins are fuller relative to item size
        ui = 1.0 / (1.0 + avg_remaining / mean_size)
        
    # Base weights for other components
    gamma_base = 0.50
    gamma_min = 0.10
    alpha_base = 0.30
    alpha_max = 0.70
    
    gamma = gamma_base - (gamma_base - gamma_min) * ui
    alpha = alpha_base + (alpha_max - alpha_base) * ui
    
    beta = 0.15
    
    epsilon_compat_base = 0.05
    epsilon_compat_max = 0.20
    epsilon_compat = epsilon_compat_base + (epsilon_compat_max - epsilon_compat_base) * ui
    
    zeta_base = 0.05
    zeta_max = 0.30
    zeta = zeta_base + (zeta_max - zeta_base) * ui
    
    # Refine Bin Age Penalty Weight:
    # Scale inversely with utilization. 
    # When utilization is high (bins full), we care less about age and more about fit.
    # When utilization is low, we care more about age to prevent fragmentation/opening new bins.
    # ui ranges [0, 1]. 
    # w_age should be high when ui is low, and low when ui is high.
    w_age_base = 0.4
    w_age_min = 0.1
    w_age = w_age_base - (w_age_base - w_age_min) * ui
    
    age_penalty = w_age * sigmoid_age_penalty

    # 5. Item Compatibility Component
    item_z = (item - mean_size) / (std_dev + epsilon)
    typicality_score = np.exp(-0.5 * item_z ** 2)
    # Re-use normalized_age_penalty here if needed, but keeping it simple for now
    compatibility_score = typicality_score * (0.5 + 0.5 * sigmoid_age_penalty)
    
    # --- Refined Component: Fragmentation Risk with Dynamic Threshold ---
    min_likely_item_size = mean_size - 2.0 * std_dev
    safe_min_size = max(min_likely_item_size, epsilon)
    gap_margin = remaining_after - safe_min_size
    deficit = np.where(gap_margin < 0, -gap_margin, 0.0)
    frag_scale = safe_min_size + epsilon
    fragmentation_risk_score = np.clip(deficit / frag_scale, 0.0, 1.0)
    
    # --- New Component: Fragility Index ---
    # Calculate the ratio of the current item's size to the minimum remaining capacity
    min_remaining = np.min(bins)
    fragility_index = item / (min_remaining + epsilon)
    
    # In tight packing regimes (high fragility_index), we want to be more careful with Best-Fit
    # to avoid creating small, unusable gaps in bins that might be needed for larger items later.
    # We apply a penalty based on the fragility index when the remaining capacity is small.
    
    # Define a threshold for "tight" regime
    fragility_threshold = 0.5
    # Smooth transition using sigmoid
    tightness_factor = 1.0 / (1.0 + np.exp(-10.0 * (fragility_index - fragility_threshold)))
    
    # Penalty for leaving a small gap when the system is tight
    # We penalize bins where remaining_after is small relative to the mean item size
    # This encourages filling bins more completely when the system is under pressure
    small_gap_penalty = np.where(remaining_after < mean_size, remaining_after / (mean_size + epsilon), 0.0)
    small_gap_penalty = small_gap_penalty * tightness_factor
    
    # Weight for the new fragility-based penalty
    omega_fragility = 0.2 * tightness_factor

    # --- Strategic Preservation Component ---
    # Goal: Prioritize bins that leave a residual capacity close to the estimated mode of future items.
    # This preserves "high-probability slots" for future arrivals.
    
    # Calculate distance of remaining_after from the estimated mode
    diff_from_mode = np.abs(remaining_after - mode_size)
    
    # Gaussian kernel for match score. The width of the Gaussian depends on std_dev.
    # If std_dev is small (consistent items), be more picky about the match.
    # If std_dev is large (variable items), be more forgiving.
    strategic_std = max(std_dev, epsilon)
    strategic_match_score = np.exp(-0.5 * (diff_from_mode / (strategic_std + epsilon)) ** 2)
    
    # Weight for Strategic Preservation
    # Higher weight when mode confidence is high and variance is low (predictable environment)
    base_strategic_w = 0.3
    strategic_w = base_strategic_w * mode_confidence
    
    # Additionally, reduce strategic weight if the item itself is very small relative to mode
    # (don't use small items to reserve huge slots if the slot is much larger than needed)
    # Actually, the match score handles this via the Gaussian.
    
    strategic_preservation_score = strategic_w * strategic_match_score
    
    # Calculate priorities
    # residual_utility and best_fit are positive contributions
    # dead_zone_penalty, age_penalty, fragmentation_risk, small_gap_penalty are negative contributions
    # compatibility_score is positive contribution
    # strategic_preservation_score is positive contribution (bonus for strategic fit)
    
    priorities = alpha * residual_utility_score + \
                 gamma * best_fit_score - \
                 beta * penalty_scale * dead_zone_penalty - \
                 age_penalty + \
                 epsilon_compat * compatibility_score - \
                 zeta * fragmentation_risk_score - \
                 omega_fragility * small_gap_penalty + \
                 strategic_preservation_score
                 
    return priorities

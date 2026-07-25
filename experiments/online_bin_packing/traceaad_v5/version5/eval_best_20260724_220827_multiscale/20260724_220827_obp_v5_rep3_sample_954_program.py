import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    ratios = item / bins
    
    # Compute log2 of ratios for items that fit
    # Add a small epsilon to avoid log2(0) if ratio is 0 (though item > 0 usually)
    log_ratios = np.log2(ratios + 1e-12)  
    
    # --- Synthesis: Power-of-Two Harmonic Scoring (from Reference p792) ---
    # Use cos(pi * log2(ratio))^2 to peak at integer log2 values.
    # This corresponds to ratios of 1, 1/2, 1/4, 1/8, etc.
    # This aligns priority with exact dyadic fractions for better structural fit.
    harmonic_scores = np.cos(np.pi * log_ratios) ** 2
    
    # --- Transfer: Utilization Gradient Term (from Reference p922) ---
    # Adds a smooth oscillatory component using sin(pi * log2(ratio)).
    # Weighted by utilization (1 - ratio) to reward bins where residual is significant.
    # This helps break ties and plateaus by considering the phase of the dyadic fit.
    utilization = 1.0 - ratios
    oscillatory_component = utilization * np.sin(np.pi * log_ratios)
    alpha = 0.2  # Scaling factor for the gradient term
    utilization_gradient = alpha * oscillatory_component
    
    # Add a small component based on raw ratio to break ties and prefer tighter fits
    # Higher ratio (tighter fit) should get a slight boost
    ratio_bias = 0.5 * ratios
    
    # Add Best Fit bonus: strongly favor bins that leave minimal residual space
    # Using squared residual to more aggressively penalize loose fits
    # Synthesized dynamic epsilon scaling from reference (10*item) for better gradient stability
    residual = bins - item
    dynamic_epsilon = 1e-9 + 10 * item
    best_fit_bonus = 1.0 / (residual**2 + dynamic_epsilon)
    
    # --- Synthesis: Smooth Waste-Split Modifier (Primary) ---
    # Implement smooth linear interpolation over [0.5*item, 2.0*item]
    # Bonus: up to +10% for residual < 0.5*item
    # Penalty: up to -10% for residual > 2.0*item
    
    threshold_low = 0.5 * item
    threshold_high = 2.0 * item
    
    # Smooth bonus: linearly decreases from 0.1 at residual=0 to 0 at residual=threshold_low
    # Using maximum to prevent negative contributions from this specific term acting as penalty
    bonus_factor = np.maximum(0.0, 0.1 * (1.0 - residual / threshold_low))
    
    # Smooth penalty: linearly increases from 0 at residual=threshold_high
    # We cap the penalty effect implicitly by the logic, though the prompt implies linear growth
    # The reference p210 logic was: penalty_factor = 0.1 * (residual / (2.0 * item) - 1.0)
    # We ensure it doesn't go below 0
    penalty_factor = np.maximum(0.0, 0.1 * (residual / threshold_high - 1.0))
    
    efficiency_modifier = 1.0 + bonus_factor - penalty_factor
    
    # --- Synthesis: Adaptive Fragmentation Penalty (Refined) ---
    # Replace fixed sigmoid with adaptive fragmentation penalty using dynamic threshold based on residual/item ratio.
    # Use refined threshold = 0.5 + 0.15 * log2(bins/item).
    # This adapts risk assessment to fit tightness with increased sensitivity (0.15 vs 0.12), 
    # avoiding overly aggressive penalties for medium fits while punishing loose ones more sharply.
    # Retain multiplicative structure and best-fit bonus.
    
    # Normalized residual for sigmoid input
    x = residual / (item + 1e-12)
    
    # Adaptive threshold based on bin tightness (bins/item)
    # log2(bins/item) = -log2(ratio)
    log_bins_item = np.log2(bins / item + 1e-12)
    dynamic_threshold = 0.5 + 0.15 * log_bins_item
    
    # Fixed steepness k=12 to maintain sharp discrimination
    k = 12.0
    exponent = k * (x - dynamic_threshold)
    frag_penalty = 0.2 / (1.0 + np.exp(exponent))
    
    # frag_penalty ranges from ~0.2 to ~0.
    frag_modifier = 1.0 - frag_penalty
    
    # --- Transfer: Multiplicative Fragmentation Logic (Reference e615 / Primary p748) ---
    # Combine harmonic structure, utilization gradient, and ratio bias.
    # Apply efficiency modifier.
    # Multiply by fragmentation modifier to decouple structural fit from risk.
    
    base_priority = harmonic_scores + utilization_gradient + ratio_bias
    priorities = base_priority * efficiency_modifier * frag_modifier
    
    # Add best fit bonus additively to maintain scale and preference for tight fits
    priorities += best_fit_bonus
    
    return priorities

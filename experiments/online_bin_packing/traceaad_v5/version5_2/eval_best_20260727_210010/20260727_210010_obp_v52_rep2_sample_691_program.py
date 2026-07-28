import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Cast bins to float64 before subtraction to ensure floating-point output.
    bins_float = bins.astype(np.float64)
    remaining = bins_float - item
    
    # Avoid division by zero or invalid modulo if item is 0.
    if item <= 0:
        # Fallback to Best Fit (minimize remaining space) if item size is invalid
        # Apply FFD bias here as well for consistency
        # Increased FFD bias coefficient to 1e-5
        return -remaining + 1e-5 * bins_float
    
    # Calculate modulo: how much remains after dividing by item size
    mod = remaining % item
    
    # Entropy-based fragmentation penalty: mod * (item - mod) / scale^2
    # This penalizes splits that create two unequal, non-multiples of the item size more harshly.
    # It is zero when mod is 0 (exact fit) or mod is item (empty bin relative to item scale),
    # and maximal when mod is item/2 (worst fragmentation).
    # Retain bins_float + 1e-9 for normalization stability as confirmed by Global Experience #2.
    scale = bins_float + 1e-9
    scale_sq = scale ** 2
    
    entropy_penalty = mod * (item - mod) / scale_sq
    
    # Weight the fragmentation term by inverse bin utilization (item / bins_float).
    # This penalizes fragmentation more heavily in bins that are less full, 
    # encouraging consolidation into fewer, fuller bins.
    utilization_weight = item / scale
    
    # Primary: Entropy-based fragmentation scaled by utilization
    # Secondary: Dynamic Best-Fit penalty scaled by item size
    # Increased coefficient to 1e-5 to strengthen the secondary tie-breaker for space conservation.
    dynamic_beta = item * 1e-5
    
    # Tertiary: First-Fit Decreasing bias
    # Increased coefficient from 1e-6 to 1e-5 to more strongly favor smaller bins.
    ffd_bias = 1e-5 * bins_float
    
    priority = -entropy_penalty * utilization_weight - dynamic_beta * remaining + ffd_bias
    
    return priority

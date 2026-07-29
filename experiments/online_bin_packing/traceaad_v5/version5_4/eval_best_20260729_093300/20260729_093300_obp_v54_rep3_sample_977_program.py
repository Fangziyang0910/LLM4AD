import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Convert bins to float to avoid integer division issues and ensure float output
    bins_float = bins.astype(np.float64)
    
    # Compute remaining capacity after placing the item
    remaining = bins_float - item
    
    # Identify feasible bins (where item fits)
    feasible = remaining >= 0
    
    # Initialize priorities with a very low value for infeasible bins
    priorities = np.full_like(bins_float, -np.inf)
    
    if np.any(feasible):
        # Use a small constant to avoid division by zero
        small_constant = 1e-9
        
        # Get indices of feasible bins for age penalty calculation
        feasible_indices = np.where(feasible)[0]
        
        # Base priority: inverse of remaining space (favor smaller remainders -> Best Fit)
        # Apply non-linear weighting by raising to power 1.5 to amplify preference for tight fits.
        # This sharpens the distinction between high-quality and low-quality fits.
        base_priority = np.power(1.0 / (remaining[feasible] + small_constant), 1.5)
        
        # Dynamic waste potential penalty
        # Calculate how well the remaining space can be filled by future items of size 'item'.
        # We look at remaining % item. If remaining is a multiple of item, waste is 0.
        # We normalize this by the remaining space to handle scale.
        
        if item <= 0:
            item_mod = np.zeros_like(remaining[feasible])
            relative_waste = np.zeros_like(remaining[feasible])
            fragmentation_penalty = np.zeros_like(remaining[feasible])
            first_fit_decrement_penalty = np.zeros_like(remaining[feasible])
        else:
            item_mod = np.fmod(remaining[feasible], item)
            # Relative waste: ratio of unusable space (modulo part) to total remaining space
            # This penalizes bins that leave fragmented space not divisible by the item size
            # Squared to aggressively penalize poor fits (waste saturation)
            relative_waste_term = item_mod / (remaining[feasible] + small_constant)
            relative_waste = relative_waste_term ** 2
            
            # Absolute fragmentation penalty
            # Penalizes bins that leave small, unusable slivers regardless of bin scale.
            # Scale is inversely proportional to the square of item size to normalize the impact.
            fragmentation_penalty = (0.01 / (item ** 2)) * item_mod
            
            # First Fit Decrement Penalty
            # Penalizes placing items into large, empty bins.
            # Dynamic scale inversely proportional to utilization gradient.
            # This causes the penalty to intensify as the bin becomes more utilized.
            
            # Utilization gradient calculation (needed for both penalty and bonus)
            bins_safe = np.where(bins_float[feasible] == 0, small_constant, bins_float[feasible])
            utilized_space = bins_float[feasible] - remaining[feasible]
            utilization_gradient = utilized_space / bins_safe
            
            # Dynamic scale: inversely proportional to utilization gradient
            first_fit_scale = 1.0 / (utilization_gradient + small_constant)
            first_fit_decrement_penalty = first_fit_scale * bins_float[feasible] / (item ** 2 + small_constant)
        
        # Utilization gradient bonus
        # Encourages consolidation of items into partially filled bins to minimize total bin count.
        # Utilization = (bins_float - remaining) / bins_float
        # Handle bins with 0 capacity to avoid division by zero
        
        if item > 0:
            # Utilization gradient calculated above
            # Dynamic utilization scale: increases as fragmentation (item_mod) decreases.
            # This creates a functional interaction where the "closing bin" incentive is strongest
            # for bins that are both well-utilized and have minimal fragmentation relative to the item size.
            utilization_scale = 0.1 / (item_mod + small_constant)
            utilization_bonus = utilization_scale * utilization_gradient
        else:
            utilization_bonus = np.zeros_like(remaining[feasible])

        # Bin age penalty
        # Penalize older bins (higher indices) to break symmetries and prevent over-packing of early bins.
        # This encourages using newer bins when other heuristic scores are tied.
        # Replace static age_penalty_scale with dynamic scale inversely proportional to base_priority
        # making the tie-breaking penalty weaker when fit quality is exceptionally high.
        base_priority_mean = np.mean(base_priority)
        dynamic_age_scale = 0.05 * base_priority_mean / (base_priority + small_constant)
        age_penalty = dynamic_age_scale * feasible_indices.astype(np.float64)
        
        # Fragmentation ratio cap (Dynamic)
        # Sets the priority of bins with remaining capacity less than a threshold (e.g., item size / 2)
        # to a significantly lower value than the base best-fit score.
        # The penalty is proportional to the ratio of wasted space (remaining capacity) to item size.
        threshold = item * 0.5
        fragmentation_cap_mask = remaining[feasible] < threshold
        
        # Dynamic penalty proportional to ratio of wasted space to item size
        # We calculate the ratio of the remaining space to the item size.
        # Scale is inversely proportional to the square root of base priority to aggressively penalize
        # bins that leave unusable slivers when the base fit quality is exceptionally high.
        if item > 0:
            waste_ratio = remaining[feasible] / (item + small_constant)
            # Apply penalty only to bins below the threshold
            fragmentation_cap_penalty = np.zeros_like(remaining[feasible])
            # Dynamic scale factor: inversely proportional to sqrt(base_priority)
            dynamic_frag_scale = 50.0 / np.sqrt(base_priority + small_constant)
            fragmentation_cap_penalty[fragmentation_cap_mask] = dynamic_frag_scale[fragmentation_cap_mask] * waste_ratio[fragmentation_cap_mask]
        else:
            fragmentation_cap_penalty = np.zeros_like(remaining[feasible])
            
        # Subtract the waste penalty and fragmentation penalty and add the utilization bonus from the base priority
        # Higher priority is better. Base priority is high for small remaining (good).
        # Waste penalty is high for bad modulo fit (bad), so we subtract it.
        # Fragmentation penalty is high for large absolute remainders (bad), so we subtract it.
        # Utilization bonus is high for highly utilized bins (good), so we add it.
        # Age penalty is high for older bins (bad), so we subtract it.
        # Fragmentation cap penalty is applied to bins with very small remaining space (bad), so we subtract it.
        # First fit decrement penalty is high for large bins (bad for large items), so we subtract it.
        priorities[feasible] = base_priority - relative_waste - fragmentation_penalty + utilization_bonus - age_penalty - fragmentation_cap_penalty - first_fit_decrement_penalty
    
    return priorities

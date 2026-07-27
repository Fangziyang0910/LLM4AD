import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    epsilon = 1e-10
    
    # Ensure we work with float to avoid integer division issues
    bins_float = bins.astype(np.float64)
    item_float = float(item)
    
    # Remaining capacity after placing the item
    remaining = bins_float - item_float
    
    # Primary score: Use 1.0 / (remaining + epsilon) to prioritize bins with smallest remaining capacity (Best Fit).
    primary_score = 1.0 / (remaining + epsilon)
    
    # Conditional smooth penalty: -(1 - remaining/item)^2 only when remaining < 2 * item
    # This reduces penalty noise for bins with larger excess capacity
    ratio = remaining / item_float
    quadratic_penalty = -(1.0 - ratio)**2
    
    # Create a mask: penalty applies only when remaining < 2 * item
    condition = remaining < 2.0 * item_float
    smooth_penalty = np.where(condition, quadratic_penalty, 0.0)
    
    # Fragment usability bonus: Strongly prioritize bins where remaining capacity is an integer multiple of the item size.
    # Calculate distance to nearest multiple of item size
    mod = remaining % item_float
    dist_to_multiple = np.minimum(mod, item_float - mod)
    
    # Inverse-square bonus: higher bonus when closer to a multiple (distance near 0)
    # Adding epsilon to avoid division by zero
    fragment_bonus = 1.0 / (dist_to_multiple**2 + epsilon)
    
    # Combine primary score, smooth penalty, and fragment bonus
    final_priority = primary_score + smooth_penalty + fragment_bonus
    
    return final_priority

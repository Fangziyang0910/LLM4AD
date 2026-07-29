import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Convert bins to float to allow floating-point operations
    bins_f = bins.astype(np.float64)
    
    # After placing the item, the remaining capacity in each bin would be:
    remaining_after = bins_f - item
    
    # We want to minimize wasted space, so higher priority for bins where remaining_after is small (but non-negative)
    # Since bins are already feasible (remaining >= 0), we can use -remaining_after as a base priority
    # This gives higher priority to bins that have less leftover space after placing the item
    
    base_priority = -remaining_after
    
    # To break ties and improve overall bin utilization, we can add a small bonus based on the bin's 
    # original capacity. Larger bins might be more flexible, but we primarily care about fit.
    # A simple approach: just use the negative remaining space as priority.
    # However, to handle edge cases and ensure finite values, we return base_priority directly.
    
    return base_priority

import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # bins contains remaining capacities of feasible bins (i.e., bins where bins[i] >= item)
    # We want to minimize number of bins, so we prefer packing items tightly.
    # Primary: smaller remaining capacity after placing item (i.e., bins[i] - item is small)
    # This is equivalent to preferring bins with smaller current remaining capacity.
    # Secondary: if tied, prefer bins with larger remaining capacity to keep them flexible.
    
    # Convert to float to avoid integer overflow/underflow issues with penalties
    bins_float = bins.astype(float)
    
    # Primary score: negative of remaining capacity after placement (higher is better when remainder is small)
    # We want small remainder, so higher priority for smaller bins[i]
    primary = bins_float  # Larger bins_float means more remaining, which we actually want to avoid for tight packing
    # Actually, for First Fit style: prefer the first bin that fits. But for Best Fit: prefer the bin that leaves least slack.
    # Best Fit: minimize (bins[i] - item), i.e., minimize remaining capacity after placement.
    # So priority should be higher when (bins[i] - item) is smaller.
    # Equivalently, priority = -(bins[i] - item) = item - bins[i], but this would make larger bins[i] have lower priority.
    # Let's use: priority = 1 / (bins[i] - item + eps) to make small remainder have high priority
    
    remainder = bins_float - item
    eps = 1e-10
    primary_score = 1.0 / (remainder + eps)
    
    # Secondary: among bins with similar remainder, prefer those with larger original capacity
    # to keep them available for larger items. But since these are remaining capacities, 
    # a larger bins[i] with same remainder means the bin was originally larger and had more space used.
    # Actually, let's keep it simple: just use best fit (minimize remainder).
    
    return primary_score

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    # Identify which bins can fit the item
    can_fit = bins >= item
    
    # Initialize priorities to -inf
    priorities = np.full_like(bins, -np.inf, dtype=float)
    
    # For bins that can fit, calculate priority as quintic inverse of remaining capacity
    # Higher priority for bins with less remaining capacity (i.e., fuller bins)
    if np.any(can_fit):
        # Calculate remaining capacity after adding item
        remaining = bins[can_fit] - item
        
        # Add a small epsilon to avoid division by zero if remaining is 0
        epsilon = 1e-8
        base_prio = 1.0 / (remaining + epsilon)**5
        
        # Add logarithmic compactness bonus to favor bins that are already partially filled
        # Increased coefficient to 0.3 to encourage tighter packing
        compactness_bonus = 0.3 * np.log1p(bins[can_fit])
        
        # Get the indices of bins that can fit the item
        indices = np.arange(len(bins))[can_fit]
        
        # Subtract linear index penalty to prefer lower-indexed bins (First-Fit tie-breaking)
        # Modified coefficient from -0.05 to -0.02 to weaken tie-breaking behavior
        prio_values = base_prio + compactness_bonus - 0.02 * indices
        
        # Assign back to the appropriate positions
        priorities[can_fit] = prio_values
    
    return priorities

import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    if not bins.size:
        return np.array([], dtype=np.float64)
    
    # Cast bins to float for computation
    bins_float = bins.astype(np.float64)
    
    # Remaining capacity after placing the item
    remaining = bins_float - item
    
    # Best Fit heuristic base: minimize waste by choosing the bin with smallest remaining capacity.
    # We return negative remaining capacity so that the bin with smallest remaining 
    # capacity has the highest priority (since we maximize priority).
    # Add capacity bias: prioritize placing items in larger bins to preserve smaller, more flexible bins.
    score = -remaining + bins_float
    
    # Dynamic fragmentation penalty based on bin count
    n_bins = len(bins)
    
    # Use exponential saturation functions for smoother adaptation
    exp_factor = 1 - np.exp(-n_bins / 50.0)
    scale_lower = 0.05 - 0.03 * exp_factor
    # Tightened scale_upper growth factor from 0.2 to 0.1 for a more conservative penalty zone
    scale_upper = 0.6 + 0.1 * exp_factor
    
    # Define the boundaries
    lower_bound = scale_lower * item
    upper_bound = scale_upper * item
    
    # Identify bins where remaining capacity falls in the fragmentation zone
    in_zone = (remaining >= lower_bound) & (remaining <= upper_bound)
    
    # Constant penalty scale as per reference program
    penalty_scale = 1.0
    
    # Apply a cubic penalty based on the cube of the residual capacity (distance from zero)
    # This provides a sharper gradient than quadratic, penalizing larger residuals more heavily
    # within the awkward gap zone.
    penalty = in_zone * (remaining ** 3) * penalty_scale
    
    score = score - penalty
    
    # Apply Bin Lifetime bias: subtract bin index to penalize older bins (assuming bins are sorted by creation time)
    # This encourages filling new bins first, addressing stagnation by introducing a temporal heuristic.
    indices = np.arange(len(bins), dtype=np.float64)
    score = score - indices
    
    return score

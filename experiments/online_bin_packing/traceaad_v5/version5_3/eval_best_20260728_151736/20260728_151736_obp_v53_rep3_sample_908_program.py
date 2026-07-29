import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    bins_f = bins.astype(np.float64)
    waste = bins_f - item
    tight_condition = waste < item * 0.5
    dynamic_penalty = -waste / (bins_f + 1e-9)
    tight_score = -1.0 * waste + dynamic_penalty
    loose_score = -np.log1p(waste) + dynamic_penalty
    return np.where(tight_condition, tight_score, loose_score)

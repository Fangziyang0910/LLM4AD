import random
import math
import scipy
try:
    import torch
except Exception:
    torch = None
import numpy as np
def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns priority with which we want to add item to each bin.
    Args:
        item: Size of item to be added to the bin.
        bins: Array of capacities for each bin.
    Return:
        Array of same size as bins with priority score of each bin.
    """
    if bins.size == 0:
        return np.empty(0, dtype=float)
    
    # Calculate remaining space if item is placed in each bin
    residual_spaces = bins.astype(float) - item
    
    # Best Fit heuristic: prefer bins with smallest residual space.
    # We invert residual space (negative) so that smaller residuals have higher priority scores.
    priorities = -residual_spaces
    
    return priorities

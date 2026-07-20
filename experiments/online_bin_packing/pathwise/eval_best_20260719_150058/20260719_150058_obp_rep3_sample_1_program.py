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
    # Create a copy of the priority array initialized to zeros
    priorities = np.zeros_like(bins, dtype=np.float64)
    
    # Find indices where the item fits
    feasible_mask = bins >= item
    
    if np.any(feasible_mask):
        # Calculate remaining space after adding the item
        remaining = bins[feasible_mask] - item
        
        # Avoid division by zero by adding a small epsilon if remaining is 0
        # Or simply use 1/(remaining + epsilon) to prioritize smaller gaps
        # Here we use a large number for zero gap (perfect fit) to prioritize it highest
        # Alternatively, simple inverse: higher priority for smaller remaining space
        
        # Let's use 1 / (remaining + 1e-6) as a basic priority score
        # Smaller remaining -> Higher priority
        
        epsilon = 1e-6
        priorities[feasible_mask] = 1.0 / (remaining + epsilon)
        
    return priorities

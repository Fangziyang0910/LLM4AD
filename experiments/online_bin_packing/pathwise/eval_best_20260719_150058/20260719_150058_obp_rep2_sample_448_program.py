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
    epsilon = 1e-9
    slack = bins - item
    
    # Identify valid bins where the item fits
    valid = slack >= 0
    
    # Initialize priorities with zeros for invalid bins
    priorities = np.zeros_like(bins)
    
    if np.any(valid):
        valid_bins = bins[valid]
        valid_slack = slack[valid]
        
        # Refined Precision Bisect Resonant Parameters
        k = 0.45
        mu = 3.83
        sigma_sq = 8.5
        gamma = 1.05
        
        # Base priority: tight-fit preference via sub-linear ratio
        base_priorities = (valid_bins / (valid_slack + epsilon)) ** k
        
        # Additive Resonant Boost (Sweet-Spot Reward)
        # Symmetric Lorentzian-like function centered at mu
        boost = gamma / ((valid_slack - mu) ** 2 + sigma_sq)
        
        # Combine base tight-fit with slack optimization additively
        priorities[valid] = base_priorities + boost
        
    return priorities

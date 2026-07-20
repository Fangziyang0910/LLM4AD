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
    remaining = bins - item
    valid_mask = remaining >= 0
    
    priorities = np.zeros_like(bins, dtype=float)
    
    if not np.any(valid_mask):
        return priorities
        
    valid_remaining = remaining[valid_mask]
    
    # Dynamic epsilon for numerical stability
    epsilon = 1e-9
    eps = 1e-9 + item * 1e-10
    
    # Power-law transformation with tuned alpha = 1.05
    alpha = 1.05
    # Raw power-law scores: higher score for smaller remaining space
    raw_score = (valid_remaining + eps) ** -alpha
    
    # Adaptive sigmoid transformation
    center = np.median(raw_score)
    spread = np.std(raw_score)
    
    # Adaptive k: inversely proportional to spread
    k = 1.0 / (spread + epsilon)
    
    # Calculate difference and clip for numerical stability
    diff = k * (raw_score - center)
    diff = np.clip(diff, -500, 500)
    
    # Apply sigmoid
    priority_scores = 1.0 / (1.0 + np.exp(-diff))
    
    # Assign scores to valid bins
    priorities[valid_mask] = priority_scores
    
    return priorities

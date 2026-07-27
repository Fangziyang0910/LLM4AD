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
    bins_float = bins.astype(np.float64)
    
    # Calculate residual capacity after placing the item
    residual = bins_float - item
    
    # Define the threshold for "usable" space.
    # A residual smaller than the current item is likely unusable for any 
    # subsequent item of similar or larger size.
    threshold = item
    
    # Base priority: Negative linear penalty on residual (Best Fit).
    # Tighter fits (smaller residual) have higher priority (less negative).
    base_priority = -residual
    
    # Waste penalty: Apply a non-linear penalty to residuals below the threshold.
    # If residual < threshold, the space is considered "waste" or fragmentation.
    # We apply a quadratic penalty to aggressively discourage these small gaps.
    # If residual >= threshold, the waste penalty is 0.
    
    # Calculate the portion of residual that is considered waste (small residuals)
    # waste_amount = residual if residual < threshold else 0
    waste_amount = np.minimum(residual, threshold)
    
    # Only apply penalty if residual < threshold. 
    # If residual >= threshold, waste_amount == threshold, but we don't want to penalize.
    # Actually, the directive says: -alpha * (r < threshold ? r^2 : 0)
    # So we need a mask or conditional.
    
    is_waste = residual < threshold
    
    # Calculate quadratic penalty for waste
    # Using a moderate alpha to balance best-fit vs waste avoidance
    alpha = 1.0 
    quadratic_penalty = np.where(is_waste, alpha * (residual ** 2), 0.0)
    
    # Total priority
    # Higher priority is better.
    # Base priority favors tight fits.
    # Quadratic penalty heavily reduces priority for very tight fits that leave unusable space.
    priorities = base_priority - quadratic_penalty
    
    return priorities

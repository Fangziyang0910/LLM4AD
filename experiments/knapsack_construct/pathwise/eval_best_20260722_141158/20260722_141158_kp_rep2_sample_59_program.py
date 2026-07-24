import random
import math
import scipy
try:
    import torch
except Exception:
    torch = None
import numpy as np
from typing import List, Tuple
def select_next_item(remaining_capacity: int, remaining_items: List[Tuple[int, int, int]]) -> Tuple[int, int, int] | None:
    """
    Select the item with the highest value-to-weight ratio that fits in the remaining capacity.

    Args:
        remaining_capacity: The remaining capacity of the knapsack.
        remaining_items: List of tuples containing (weight, value, index) of remaining items.

    Returns:
        The selected item as a tuple (weight, value, index), or None if no item fits.
    """
    total_weight_of_remaining = sum(w for w, v, i in remaining_items)
    estimated_initial_capacity = remaining_capacity + total_weight_of_remaining
    
    # Avoid division by zero
    if estimated_initial_capacity == 0:
        estimated_initial_capacity = 1.0
        
    # Hyperparameters
    k = 2.0   # Power-law scaling exponent
    gamma = 0.1  # Fragmentation penalty coefficient
    beta = 0.5   # Spatial fit bonus coefficient
    
    best_item = None
    best_score = -float('inf')
    
    for weight, value, index in remaining_items:
        if weight <= remaining_capacity:
            # Calculate value density
            if weight > 0:
                density = value / weight
            else:
                density = float('inf')
            
            # Capacity ratio: 1.0 when full, 0.0 when empty (relative to estimated initial)
            capacity_ratio = remaining_capacity / estimated_initial_capacity
            
            # Calculate gap_ratio and fit_ratio
            if remaining_capacity > 0:
                gap_ratio = (remaining_capacity - weight) / remaining_capacity
                fit_ratio = 1.0 - gap_ratio  # equivalent to weight / remaining_capacity
            else:
                # If remaining_capacity is 0, weight must be 0
                gap_ratio = 0.0
                fit_ratio = 0.0
            
            if density == float('inf'):
                # If weight is 0, density is inf. Score is dominated by infinity.
                score = float('inf')
            else:
                # Hybrid score: 
                # density (value efficiency)
                # + (capacity_ratio ** k) * value (context-aware absolute value scaling)
                # - gamma * gap_ratio (penalty for leaving large gaps)
                # + beta * fit_ratio (bonus for filling space tightly)
                value_score = (capacity_ratio ** k) * value
                gap_penalty = gamma * gap_ratio
                fit_bonus = beta * fit_ratio
                
                score = density + value_score - gap_penalty + fit_bonus
            
            if score > best_score:
                best_score = score
                best_item = (weight, value, index)
                
    return best_item

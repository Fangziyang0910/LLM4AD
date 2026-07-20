import random
import math
import scipy
try:
    import torch
except Exception:
    torch = None
import numpy as np
def select_next_item(remaining_capacity: int, remaining_items: List[Tuple[int, int, int]]) -> Tuple[int, int, int] | None:
    """
    Select the item with the highest value-to-weight ratio that fits in the remaining capacity.

    Args:
        remaining_capacity: The remaining capacity of the knapsack.
        remaining_items: List of tuples containing (weight, value, index) of remaining items.

    Returns:
        The selected item as a tuple (weight, value, index), or None if no item fits.
    """
    fitting_items = [item for item in remaining_items if item[0] <= remaining_capacity]
    
    if not fitting_items:
        return None
    
    # --- 1. Robust Local State Estimation ---
    # Ground adaptation in immediate geometric constraints
    max_fitting_weight = max((w for w, v, i in fitting_items), default=0)
    if remaining_capacity > 0:
        fill_level = min(1.0, max_fitting_weight / remaining_capacity)
    else:
        fill_level = 1.0
        
    # --- 2. Scale-Aware Gamma Modulation ---
    total_fitting_weight = sum(item[0] for item in fitting_items)
    avg_weight = total_fitting_weight / len(fitting_items) if fitting_items else 1.0
    
    if avg_weight == 0:
        avg_weight = 1.0
        
    if remaining_capacity > 0:
        scale_ratio = min(1.0, avg_weight / remaining_capacity)
    else:
        scale_ratio = 1.0
        
    gamma_min = 1.2
    gamma_max = 2.5 + 0.6 * scale_ratio
    
    adaptive_gamma = gamma_min + (gamma_max - gamma_min) * fill_level
    
    # --- 3. Static Alpha Adjustment ---
    # Directive: static alpha_fit = 0.25
    # Directive: static alpha_val = 0.10 (removed dynamic logic)
    alpha_fit = 0.25
    alpha_val = 0.10

    # --- 4. Value Normalization ---
    max_fitting_value = max((item[1] for item in fitting_items), default=1)
    if max_fitting_value == 0:
        max_fitting_value = 1
    
    # --- 5. Fragmentation Penalty Setup ---
    positive_fitting_items = [item for item in fitting_items if item[0] > 0]
    if positive_fitting_items:
        min_weight = min((it[0] for it in positive_fitting_items))
        if min_weight == 0:
            min_weight = 1
    else:
        min_weight = 1
        
    # --- 6. Novel Exploration Nudge: Capacity Variance Penalty ---
    # Calculates a penalty based on the variance of weights in remaining items.
    # High variance suggests that picking an item might leave a "gap" that no 
    # other item can fill efficiently, increasing fragmentation risk.
    weight_variance = 0.0
    if len(positive_fitting_items) > 1:
        mean_w = total_fitting_weight / len(positive_fitting_items)
        variance = sum((item[0] - mean_w) ** 2 for item in positive_fitting_items) / len(positive_fitting_items)
        # Normalize variance by mean weight squared to be scale-independent
        if mean_w > 0:
            weight_variance = variance / (mean_w ** 2)
            
    def composite_score(item: Tuple[int, int, int]) -> float:
        weight, value, index = item
        
        if weight == 0:
            # Handle zero weight items: infinite density, prioritize high value
            return 1e9 * value + value
        
        ratio = value / weight
        
        # Fit factor rewards items that fill a significant portion of remaining capacity
        fit_factor = (weight / remaining_capacity) ** adaptive_gamma if remaining_capacity > 0 else 1.0
        
        # Value factor provides a slight boost for high-value items
        value_factor = 1 + alpha_val * (value / max_fitting_value)
        
        residual_space = remaining_capacity - weight
        
        fragmentation_penalty = 1.0
        if residual_space > 0 and residual_space < min_weight:
            waste_ratio = residual_space / min_weight
            fragmentation_penalty = 1.0 / (1.0 + (2.0 * (waste_ratio ** 2)))
            
        # Novel: Capacity Variance Penalty
        # Penalize selections in high-variance scenarios where picking this item 
        # might lead to unfillable gaps later, unless the item itself is very heavy (stable).
        variance_penalty = 1.0
        if weight_variance > 0.1: # Only apply if variance is significant
            # Higher penalty if the item is small relative to the mean, 
            # as small items in high-variance sets often leave weird gaps.
            size_ratio = weight / avg_weight if avg_weight > 0 else 1.0
            # Reduce score if item is small and variance is high
            variance_penalty = 1.0 / (1.0 + 0.5 * weight_variance * (1.0 / (size_ratio + 0.1)))
            
        score = ratio * (1 + alpha_fit * fit_factor) * value_factor * fragmentation_penalty * variance_penalty
        
        return score

    selected_item = max(
        fitting_items,
        key=composite_score
    )
    
    return selected_item

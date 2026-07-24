
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
    feasible_items = []
    for item in remaining_items:
        weight, value, index = item
        if weight <= remaining_capacity:
            feasible_items.append(item)
            
    if not feasible_items:
        return None
        
    # If only one item, pick it
    if len(feasible_items) == 1:
        return feasible_items[0]

    # Handle zero-weight items with positive value as infinitely good
    zero_weight_positive_value_items = [item for item in feasible_items if item[0] == 0 and item[1] > 0]
    if zero_weight_positive_value_items:
        # Among zero-weight positive value items, pick the one with highest value
        best_z = max(zero_weight_positive_value_items, key=lambda x: x[1])
        return best_z

    # Parameters for the new look-ahead scoring
    # Use a harmonic mean based normalization and a convex penalty for capacity usage
    # Logarithmic scaling to moderate sensitivity compared to power scaling
    penalty_coeff = 0.5
    log_base = 2.0
    harmonic_factor = 1.0
    
    # Calculate harmonic mean of densities for normalization among feasible items
    # Harmonic mean is more sensitive to low-density items, providing a different perspective than arithmetic mean
    densities = []
    for w, v, i in feasible_items:
        if w > 0:
            densities.append(v / w)
        else:
            # Zero weight with zero value doesn't affect harmonic mean significantly if we filter or handle separately
            # But we already handled zero weight positive value.
            # If weight is 0 and value is 0, density is 0. Harmonic mean of including 0 is 0.
            # Let's filter out zero weights for density calculation to avoid division by zero in harmonic mean
            pass
    
    if densities:
        # Harmonic mean = n / sum(1/di)
        inv_sum = sum(1.0 / d for d in densities)
        avg_density_harmonic = len(densities) / inv_sum if inv_sum > 0 else 1e-9
    else:
        avg_density_harmonic = 1e-9

    best_item = None
    best_total_est = -float('inf')

    for item in feasible_items:
        w, v, idx = item
        
        # Remaining capacity after picking this item
        rem_cap = remaining_capacity - w
        
        # Estimate value of remaining items using a fast greedy heuristic
        # Create list of other feasible items
        remaining = [it for i, it in enumerate(feasible_items) if it[2] != idx]
        
        est_value = 0
        temp_cap = rem_cap
        
        # Define the sorting key for the residual greedy pack
        def get_sort_key(it, _avg_density=avg_density_harmonic, _penalty_coeff=penalty_coeff, _log_base=log_base, _rem_cap=remaining_capacity):
            wt, vl, ind = it
            if wt == 0:
                return float('inf') if vl > 0 else -float('inf')
            
            # Normalized density ratio using harmonic mean
            density_ratio = (vl / wt) / _avg_density
            
            # Capacity fraction
            capacity_fraction = wt / _rem_cap
            
            # Non-linear convex penalty: penalty increases super-linearly with capacity usage
            # Using power of 2 for convexity, scaled by penalty_coeff
            penalty = _penalty_coeff * (capacity_fraction ** 2)
            
            # Combine density ratio and penalty
            # Score = density_ratio * (1 - penalty)
            # Ensure non-negative
            base_score = density_ratio * (1 - penalty)
            if base_score < 0:
                base_score = 0
                
            # Apply logarithmic scaling to moderate score differences
            # log_base + log(base_score) if base_score > 0 else 0
            # Using log(1 + base_score) to handle small positive scores smoothly
            if base_score > 0:
                import math
                final_score = math.log(1 + base_score, _log_base)
            else:
                final_score = 0
                
            return final_score

        # Sort remaining items by the modified density key and perform greedy pack
        for it in sorted(remaining, key=get_sort_key, reverse=True):
            wt, vl, ind = it
            if wt <= temp_cap:
                est_value += vl
                temp_cap -= wt
        
        total_est = v + est_value
        
        # Select the item with the highest estimated total value
        if total_est > best_total_est:
            best_total_est = total_est
            best_item = item
        elif total_est == best_total_est:
            # Tie-break: pick lighter weight to preserve capacity flexibility
            if w < best_item[0]:
                best_item = item
    
    return best_item

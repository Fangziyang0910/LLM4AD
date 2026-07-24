
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
    import random
    import math
    from typing import List, Tuple

    if remaining_capacity <= 0 or not remaining_items:
        return None

    # Filter feasible items
    feasible_items = [(w, v, idx) for w, v, idx in remaining_items if w > 0 and w <= remaining_capacity]
    
    if not feasible_items:
        return None

    # If few items, evaluate all. If many, sample to save time and introduce stochasticity.
    num_items = len(feasible_items)
    # Sample slightly more than No.2 to ensure better coverage, but keep it limited for efficiency
    sample_size = min(75, num_items)
    if num_items > sample_size:
        candidates = random.sample(feasible_items, sample_size)
    else:
        candidates = feasible_items

    best_item = None
    best_score = -float('inf')
    best_value = -float('inf')
    best_weight = float('inf')
    best_index = float('inf')

    # Estimate original capacity based on remaining capacity and typical item sizes
    # Use the maximum weight in candidates as a proxy for typical item size
    if candidates:
        max_weight = max(w for w, v, idx in candidates)
        # Assume original capacity is roughly remaining_capacity + some multiple of max_weight
        # This is a heuristic to estimate how "full" the knapsack is
        estimated_original_capacity = remaining_capacity + max_weight * 5
        if estimated_original_capacity <= 0:
            estimated_original_capacity = 1
        capacity_ratio_overall = remaining_capacity / estimated_original_capacity
    else:
        capacity_ratio_overall = 1.0

    # Dynamic weighting: use a logistic function to transition from value-ratio focus to fit focus
    # When remaining capacity is high (early stages), focus more on value/weight ratio
    # When remaining capacity is low (late stages), focus more on fitting items tightly
    # Logistic function: 1 / (1 + exp(-k * (x - x0)))
    k = 10.0  # Steepness of transition
    x0 = 0.5  # Midpoint of transition
    
    # Calculate dynamic weight for fit score
    logistic_val = 1 / (1 + math.exp(-k * (capacity_ratio_overall - x0)))
    fit_weight = logistic_val  # When capacity is high, fit_weight is low; when low, fit_weight is high
    
    # Exponent for fit score: adjust based on remaining capacity
    # Higher exponent when remaining capacity is small to penalize loose fits more
    fit_exponent = 2.0 + (1.0 - capacity_ratio_overall) * 2.0  # Range: 2.0 to 4.0
    
    for weight, value, index in candidates:
        if weight <= remaining_capacity:
            ratio = value / weight
            
            # Capacity utilization ratio: how much of the remaining space this item takes
            capacity_ratio = weight / remaining_capacity if remaining_capacity > 0 else 0
            
            # Fit score: rewards items that use a larger portion of the remaining capacity
            # Using a power law with dynamic exponent
            fit_score = capacity_ratio ** fit_exponent
            
            # Composite score: Ratio boosted by fit score with dynamic weighting
            # The fit_weight dynamically adjusts based on how full the knapsack is
            composite_score = ratio * (1 + fit_weight * fit_score)
            
            # Tie-breaking:
            # Primary: Higher value
            # Secondary: Lower weight
            # Tertiary: Lower index
            
            if (composite_score > best_score + 1e-9):
                best_score = composite_score
                best_value = value
                best_weight = weight
                best_index = index
                best_item = (weight, value, index)
            elif abs(composite_score - best_score) <= 1e-9:
                # Primary tie-break: Higher value
                if value > best_value:
                    best_value = value
                    best_weight = weight
                    best_index = index
                    best_item = (weight, value, index)
                elif value == best_value:
                    # Secondary tie-break: Lower weight
                    if weight < best_weight:
                        best_weight = weight
                        best_index = index
                        best_item = (weight, value, index)
                    elif weight == best_weight:
                        # Tertiary tie-break: Lower index
                        if index < best_index:
                            best_index = index
                            best_item = (weight, value, index)

    return best_item

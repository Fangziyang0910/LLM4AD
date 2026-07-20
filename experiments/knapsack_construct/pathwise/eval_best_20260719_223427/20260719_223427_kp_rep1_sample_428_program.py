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
    best_item = None
    best_score = float('-inf')
    best_compat_ratio = float('inf')
    best_max_frag_ratio = float('inf')
    best_residual_value = float('-inf')
    best_weight = float('inf')
    
    # 1. Engine Stabilization: Static Parameters and Fixed Lookahead
    ALPHA = 0.50
    BETA = 0.15
    LAMBDA = 0.4
    GAMMA = 0.1
    K = 5  # Fixed lookahead depth
    
    # Pre-filter items that fit
    fitting_items = [item for item in remaining_items if item[0] <= remaining_capacity]
    
    if not fitting_items:
        return None

    # Pre-calculate weights of all fitting items for fragmentation checks
    # Filter out zero weights to avoid modulo by zero errors later
    fitting_weights = [item[0] for item in fitting_items if item[0] > 0]

    # World-model reflection optimization: Sort fitting_items by density descending
    def density_key(item):
        w, v, _ = item
        if w == 0:
            return float('inf') if v > 0 else 0
        return v / w
    
    # Create a sorted list for iteration and optimized lookahead
    sorted_fitting_items = sorted(fitting_items, key=density_key, reverse=True)

    # Calculate max_possible_value (sum of all fitting values) for scale-invariant normalization
    max_possible_value = sum(item[1] for item in fitting_items)
    if max_possible_value == 0:
        max_possible_value = 1.0 # Avoid division by zero

    # Calculate max_fitting_value for primary score normalization
    fitting_values = [item[1] for item in fitting_items]
    max_fitting_value = max(fitting_values, default=0)
    if max_fitting_value == 0:
        max_fitting_value = 1  # Avoid division by zero

    # Iterate through fitting items to find the best one
    for item in fitting_items:
        weight, value, index = item
        
        # Handle zero-weight items
        if weight == 0:
            if value > 0:
                primary_score = float('inf')
                estimated_residual_value = 0.0
                compat_ratio = 0.0
                max_frag_ratio = 0.0
            else:
                primary_score = 0
                estimated_residual_value = 0.0
                compat_ratio = 0.0
                max_frag_ratio = 0.0
        else:
            # Primary Score Refinement: Value-Normalized Adaptive Factor
            ratio = value / weight
            
            gap = remaining_capacity - weight
            if remaining_capacity > 0:
                relative_gap = gap / remaining_capacity
                
                # Apply power term to relative gap with static ALPHA
                gap_term = relative_gap ** ALPHA
                
                # Value-Normalized Adaptive Factor from rollout_34_0_0_1
                adaptive_factor = 1.0 + BETA * (weight / remaining_capacity) * (value / max_fitting_value)
                
                # Combine gap term and adaptive factor in the denominator with static Lambda
                modulator = 1.0 / (1.0 + LAMBDA * gap_term * adaptive_factor)
            else:
                modulator = 1.0
            
            primary_score = ratio * modulator
            
            # Lookahead Integration: Fixed K=5 depth
            residual_capacity = remaining_capacity - weight
            estimated_residual_value = 0.0
            
            if residual_capacity > 0:
                count = 0
                for other_item in sorted_fitting_items:
                    if count >= K:
                        break
                    other_w, other_v, other_idx = other_item
                    if other_w <= residual_capacity:
                        estimated_residual_value += other_v
                        count += 1
            
            # Calculate Normalized Fragmentation Metrics
            compat_ratio = float('inf')
            max_frag_ratio = float('inf')

            if fitting_weights and remaining_capacity > 0:
                residual_capacity_for_frag = residual_capacity
                
                compatibility_sum = 0
                max_fragment = 0
                
                for other_w in fitting_weights:
                    rem = residual_capacity_for_frag % other_w
                    compatibility_sum += rem
                    if rem > max_fragment:
                        max_fragment = rem
                
                # Normalize Compatibility Ratio
                denominator = remaining_capacity * len(fitting_weights)
                if denominator > 0:
                    compat_ratio = compatibility_sum / denominator
                
                # Normalize Max Fragment Ratio
                if remaining_capacity > 0:
                    max_frag_ratio = max_fragment / remaining_capacity
            else:
                compat_ratio = 0.0
                max_frag_ratio = 0.0
        
        # Calculate Effective Score for Tier 1 comparison
        # Lookahead normalization uses sum of fitting values (scale-invariant)
        if weight == 0:
             if value > 0:
                 effective_score = float('inf')
             else:
                 effective_score = 0
        else:
            lookahead_term = estimated_residual_value / max_possible_value
            effective_score = primary_score + GAMMA * lookahead_term

        # Determine if this item is better than the current best
        is_better = False
        
        # Tie-Breaking Hierarchy from entail_31_2
        # Tier 1: Effective Score
        if effective_score > best_score:
            is_better = True
        elif abs(effective_score - best_score) < 1e-9:
            # Tier 2: Normalized Compatibility Ratio (Minimize)
            if compat_ratio < best_compat_ratio:
                is_better = True
            elif abs(compat_ratio - best_compat_ratio) < 1e-9:
                # Tier 3: Max Fragment Ratio (Minimize)
                if max_frag_ratio < best_max_frag_ratio:
                    is_better = True
                elif abs(max_frag_ratio - best_max_frag_ratio) < 1e-9:
                    # Tier 4: Raw Estimated Residual Value (Maximize)
                    if estimated_residual_value > best_residual_value:
                        is_better = True
                    elif abs(estimated_residual_value - best_residual_value) < 1e-9:
                        # Tier 5: Weight (Minimize) -> Index (Minimize)
                        if weight < best_weight:
                            is_better = True
                        elif abs(weight - best_weight) < 1e-9:
                            if best_item is None or index < best_item[2]:
                                is_better = True
        
        if is_better:
            best_score = effective_score
            best_item = item
            best_weight = weight
            best_compat_ratio = compat_ratio
            best_max_frag_ratio = max_frag_ratio
            best_residual_value = estimated_residual_value

    return best_item

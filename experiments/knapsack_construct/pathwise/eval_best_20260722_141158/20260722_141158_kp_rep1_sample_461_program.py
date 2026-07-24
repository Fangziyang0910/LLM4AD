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
    if not remaining_items:
        return None

    # 1. Filter valid items and compute global metrics
    valid_items = []
    global_sum_weight = 0
    global_sum_value = 0
    
    for w, v, i in remaining_items:
        if w <= remaining_capacity:
            valid_items.append((w, v, i))
            global_sum_weight += w
            global_sum_value += v
    
    if not valid_items:
        return None

    # 2. Pre-sort valid items by ratio descending for efficient greedy look-ahead
    # Sort by value/weight ratio descending. Handle weight=0 as infinite ratio.
    valid_items_sorted = sorted(
        valid_items, 
        key=lambda x: x[1]/x[0] if x[0] > 0 else float('inf'), 
        reverse=True
    )
    
    # Extract just the items for easier iteration in look-ahead
    # We keep the full tuples (w, v, i)
    
    def estimate_residual_value(capacity: int, available_items: List[Tuple[int, int, int]]) -> float:
        """
        Greedy approximation of value that can be filled in the remaining capacity.
        Uses the pre-sorted available_items (by ratio) to simulate filling.
        O(N) complexity relative to number of available items.
        """
        current_val = 0.0
        current_cap = capacity
        
        for w, v, i in available_items:
            if w <= current_cap:
                current_val += v
                current_cap -= w
            if current_cap == 0:
                break
                
        return current_val

    def calculate_score(item: Tuple[int, int, int]) -> Tuple[float, float]:
        """
        Calculates a composite score for an item.
        Returns (total_score, residual_value)
        """
        w, v, i = item
        
        # Base Immediate Score: Value density + Super-linear fill bonus
        # Fill factor: how much of the knapsack this item fills relative to capacity
        if w == 0:
            ratio = float('inf')
            fill_factor = 1.0 # Treat as perfect fill for bonus calculation? Or 0? 
            # If weight is 0, it doesn't consume capacity, so fill_factor is 0 in traditional sense,
            # but let's say it adds value for free.
            # Let's stick to standard: fill_factor = w/cap. If w=0, fill=0.
            # But ratio is inf.
            # Let's handle w=0 separately if needed, but usually w>0 in knapsack.
            # Assuming w>0 for standard cases.
            ratio = v # Placeholder if w=0, value is max per 0 weight
            fill_factor = 0.0
        else:
            ratio = v / w
            fill_factor = w / remaining_capacity
        
        # Hyperparameters
        alpha = 0.5      # Fill bonus influence
        exponent = 2.0   # Super-linear exponent
        
        immediate_score = ratio + alpha * (fill_factor ** exponent)
        
        # Weight Homogeneity Penalty
        # Penalize deviation from global mean weight to encourage diverse packing
        if global_sum_weight > 0 and len(valid_items) > 1:
            mean_w = global_sum_weight / len(valid_items)
            rel_dev = abs(w - mean_w) / mean_w
            penalty = 0.9 * (rel_dev ** 2)
            immediate_score -= penalty

        # Look-Ahead Residual Value
        # Estimate how much additional value can be squeezed in after picking this item
        residual_cap = remaining_capacity - w
        
        # Filter items that fit in residual capacity
        # We can reuse valid_items_sorted, but must skip the current item
        residual_candidates = [
            (ow, ov, oi) for ow, ov, oi in valid_items_sorted 
            if oi != i and ow <= residual_cap
        ]
        
        residual_value = estimate_residual_value(residual_cap, residual_candidates)
        
        # Total Score: Immediate Value + Discounted Residual Value
        # Discount factor gamma < 1 to prioritize immediate high-value picks 
        # but heavily weigh potential future accumulation
        gamma = 0.8
        
        total_score = v + gamma * residual_value
        
        return total_score, residual_value

    # 3. Evaluate all valid candidates
    best_total_score = -1.0
    best_item = None
    
    for item in valid_items:
        score, res_val = calculate_score(item)
        
        # Selection Criteria: Maximize Total Score
        # Tie-breaker: Higher residual value (more space utilization potential)
        # Tie-breaker 2: Lower weight (leaves more absolute capacity)
        
        if score > best_total_score:
            best_total_score = score
            best_item = item
        elif abs(score - best_total_score) < 1e-9:
            # Tie-breaking
            current_best_score, current_best_res = calculate_score(best_item)
            if res_val > current_best_res:
                best_item = item
            elif res_val == current_best_res:
                if item[0] < best_item[0]:
                    best_item = item

    return best_item

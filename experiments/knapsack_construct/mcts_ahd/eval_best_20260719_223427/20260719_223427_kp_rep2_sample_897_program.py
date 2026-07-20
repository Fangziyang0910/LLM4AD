
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
    from typing import List, Tuple
    import math

    # Filter items that fit within the remaining capacity
    feasible_items = [item for item in remaining_items if item[0] <= remaining_capacity]
    
    if not feasible_items:
        return None
    
    # If only one item is feasible, pick it
    if len(feasible_items) == 1:
        return feasible_items[0]

    # Separate zero-weight items
    zero_weight_items = [item for item in feasible_items if item[0] == 0]
    non_zero_weight_items = [item for item in feasible_items if item[0] > 0]
    
    # If there are zero-weight items, pick the one with highest value
    if zero_weight_items:
        return max(zero_weight_items, key=lambda item: item[1])
    
    if not non_zero_weight_items:
        return None

    # Calculate modified efficiencies for non-zero weight items
    # Using alpha = 0.6 to favor lighter items more strongly than the original 0.8
    alpha = 0.6
    items_with_efficiency = [(item, item[1] / (item[0] ** alpha)) for item in non_zero_weight_items]
    
    # Find percentiles for dynamic threshold
    efficiencies = sorted([ratio for _, ratio in items_with_efficiency])
    n = len(efficiencies)
    # Use 80th percentile as threshold (stricter than 75th)
    idx_80 = int(n * 0.80)
    threshold = efficiencies[min(idx_80, n - 1)]
    
    # Filter items that meet the dynamic high-efficiency threshold
    high_efficiency_items = [(item, eff) for item, eff in items_with_efficiency if eff >= threshold]
    
    # If no items meet threshold, consider all
    if not high_efficiency_items:
        high_efficiency_items = items_with_efficiency

    def simulate_greedy_fill(capacity: int, items: List[Tuple[int, int, int]], alpha: float) -> float:
        """
        Simulates a greedy fill of the knapsack with given items and capacity using modified efficiency.
        Returns the total value obtained.
        """
        if capacity <= 0 or not items:
            return 0.0
        
        # Sort items by modified efficiency descending for greedy selection
        candidates = sorted(items, key=lambda x: x[1] / (x[0] ** alpha) if x[0] > 0 else float('inf'), reverse=True)
        
        current_val = 0.0
        current_cap = capacity
        
        for item in candidates:
            w, v, idx = item
            if w <= current_cap:
                current_val += v
                current_cap -= w
            else:
                # For 0/1 knapsack, we skip if it doesn't fit
                pass
                
        return current_val

    def calculate_hybrid_score(item_tuple):
        item, eff = item_tuple
        weight, value, index = item
        
        if weight == 0:
            return value
        
        # 1. Base Score: Modified Efficiency (from threshold logic)
        
        # 2. Weight Ratio Penalty (Quadratic penalty inspired by No.2 but applied differently)
        # Penalties heavier items quadratically relative to remaining capacity
        weight_ratio = weight / remaining_capacity
        # Quadratic penalty: 1 / (1 + weight_ratio^2)
        penalty_factor = 1.0 / (1.0 + weight_ratio ** 2)
        
        # 3. Lookahead Marginal Gain
        # Scenario A: Pick current item, then greedy fill the rest using modified efficiency
        cap_after_pick = remaining_capacity - weight
        # Items available after picking current one (excluding current)
        # Filter ensures items fit in the reduced capacity
        items_after_pick = [i for i in feasible_items if i[2] != index and i[0] <= cap_after_pick]
        val_if_picked = value + simulate_greedy_fill(cap_after_pick, items_after_pick, alpha)
        
        # Scenario B: Do not pick current item, greedy fill with full capacity (excluding current)
        items_no_pick = [i for i in feasible_items if i[2] != index]
        val_if_not_picked = simulate_greedy_fill(remaining_capacity, items_no_pick, alpha)
        
        # Marginal gain
        marginal_gain = val_if_picked - val_if_not_picked
        
        # Normalize marginal gain by weight
        normalized_marginal = marginal_gain / weight if weight > 0 else 0.0
        
        # Clamp normalized marginal to avoid extreme scores
        normalized_marginal = max(-1.0, min(1.0, normalized_marginal))
        
        # Hybrid Score: Modified Efficiency * Penalty Factor * (1 + Normalized Marginal Gain)
        score = eff * penalty_factor * (1.0 + normalized_marginal)
        
        return score

    # Select the item that maximizes the hybrid score among high-efficiency candidates
    best_item_pair = max(high_efficiency_items, key=calculate_hybrid_score)
    result = best_item_pair[0]
    return result

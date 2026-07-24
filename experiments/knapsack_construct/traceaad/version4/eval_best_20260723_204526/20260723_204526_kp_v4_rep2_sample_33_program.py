from typing import List, Tuple, Optional

def select_next_item(remaining_capacity: int, remaining_items: List[Tuple[int, int, int]]) -> Tuple[int, int, int] | None:
    """
    Select the item with the highest value-to-weight ratio that fits in the remaining capacity.

    Args:
        remaining_capacity: The remaining capacity of the knapsack.
        remaining_items: List of tuples containing (weight, value, index) of remaining items.

    Returns:
        The selected item as a tuple (weight, value, index), or None if no item fits.
    """
    # Filter items that fit in the remaining capacity
    fitting_items = [(w, v, i) for w, v, i in remaining_items if w <= remaining_capacity]
    
    if not fitting_items:
        return None
    
    if len(fitting_items) == 1:
        return fitting_items[0]
    
    # For each fitting item, estimate the future value we can get from the remaining capacity
    # after selecting this item using exact 0/1 Knapsack DP
    def estimate_future_value(selected_item: Tuple[int, int, int], remaining_cap: int) -> int:
        """Estimate the exact maximum value of items that can be packed in the remaining capacity 
        after selecting selected_item using 0/1 Knapsack DP."""
        w_sel, v_sel, i_sel = selected_item
        left_cap = remaining_cap - w_sel
        
        # If no capacity left, future value is 0
        if left_cap <= 0:
            return 0
        
        # Get all other items (excluding the selected one)
        others = [(w, v, i) for w, v, i in remaining_items if (w, v, i) != selected_item]
        
        # Filter others that actually fit in left_cap to reduce DP size
        feasible_others = [(w, v, i) for w, v, i in others if w <= left_cap]
        
        if not feasible_others:
            return 0
        
        # Use 0/1 Knapsack DP to find exact maximum value
        # dp[w] = max value achievable with capacity w
        dp = [0] * (left_cap + 1)
        
        for w, v, i in feasible_others:
            # Iterate backwards to ensure each item is used at most once
            for cap in range(left_cap, w - 1, -1):
                if dp[cap - w] + v > dp[cap]:
                    dp[cap] = dp[cap - w] + v
        
        return dp[left_cap]
    
    best_item = None
    best_score = -1
    
    for item in fitting_items:
        future_val = estimate_future_value(item, remaining_capacity)
        score = item[1] + future_val  # Own value + estimated future value
        
        if score > best_score:
            best_score = score
            best_item = item
        elif score == best_score and best_item is not None:
            # Tie-break: prefer higher value, then lower weight
            if item[1] > best_item[1]:
                best_item = item
            elif item[1] == best_item[1] and item[0] < best_item[0]:
                best_item = item
    
    return best_item

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
    
    # Filter items that fit in remaining capacity
    feasible_items = [item for item in remaining_items if item[0] <= remaining_capacity]
    
    if not feasible_items:
        return None
    
    def dp_knapsack(capacity: int, items: List[Tuple[int, int, int]]) -> float:
        """
        Compute the exact maximum value that can be packed into the given capacity
        using the given items via 0/1 knapsack dynamic programming.
        """
        if capacity <= 0 or not items:
            return 0.0
        
        # dp[w] = max value achievable with capacity w
        dp = [0.0] * (capacity + 1)
        
        for weight, value, idx in items:
            # Iterate backwards to ensure each item is used at most once
            for w in range(capacity, weight - 1, -1):
                if dp[w - weight] + value > dp[w]:
                    dp[w] = dp[w - weight] + value
        
        return dp[capacity]
    
    best_item = None
    best_estimated_total = -1
    
    for item in feasible_items:
        weight, value, idx = item
        # Calculate remaining capacity after selecting this item
        rem_cap = remaining_capacity - weight
        
        # Get items other than the current candidate
        other_items = [it for it in remaining_items if it[2] != idx]
        
        # Compute the exact optimal value we can get from the remaining capacity with other items
        estimated_remaining = dp_knapsack(rem_cap, other_items)
        
        # Total estimated value if we pick this item
        total_estimated = value + estimated_remaining
        
        if total_estimated > best_estimated_total:
            best_estimated_total = total_estimated
            best_item = item
    
    return best_item


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
    if not remaining_items or remaining_capacity <= 0:
        return None

    # Filter items that fit in the current remaining capacity
    feasible_items = [item for item in remaining_items if item[0] <= remaining_capacity]
    
    if not feasible_items:
        return None

    def knapsack_dp(capacity: int, items: List[Tuple[int, int, int]]) -> int:
        """
        Computes the maximum value achievable for a given capacity and set of items
        using a standard 0/1 Knapsack DP approach.
        """
        # dp[w] = max value achievable with capacity w
        dp = [0] * (capacity + 1)
        
        for w_item, v_item, idx in items:
            # Iterate backwards to avoid using the same item multiple times
            for w in range(capacity, w_item - 1, -1):
                if dp[w - w_item] + v_item > dp[w]:
                    dp[w] = dp[w - w_item] + v_item
                    
        return dp[capacity]

    best_item = None
    best_total_value = -1

    # For each feasible item, calculate the total value if we pick it
    # Total Value = Value of current item + Optimal Value of remaining items with remaining capacity
    for item in feasible_items:
        w_item, v_item, idx = item
        
        # Remaining capacity if we pick this item
        new_capacity = remaining_capacity - w_item
        
        # Remaining items if we pick this item (exclude current item)
        # Note: remaining_items might contain duplicates of indices if not careful, 
        # but typically in greedy steps, we remove the selected item.
        # Here we simulate the state after picking 'item'.
        remaining_after_pick = [i for i in remaining_items if i[2] != idx]
        
        # Calculate the optimal value for the rest of the problem
        # To keep computation feasible, we might need to limit the scope or use a heuristic if items are many.
        # However, for the purpose of this "novel algorithm" demonstration, we use exact DP on the subset.
        # Optimization: If remaining_after_pick is large, this is O(N*Capacity).
        # Given the context of single-step selection, this is a valid "lookahead" strategy.
        
        future_value = knapsack_dp(new_capacity, remaining_after_pick)
        
        total_value = v_item + future_value
        
        if total_value > best_total_value:
            best_total_value = total_value
            best_item = item

    return best_item

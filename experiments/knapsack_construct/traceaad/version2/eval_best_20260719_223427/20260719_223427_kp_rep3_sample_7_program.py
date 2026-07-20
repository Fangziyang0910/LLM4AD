from typing import List, Tuple, Optional

def select_next_item(remaining_capacity: int, remaining_items: List[Tuple[int, int, int]]) -> Tuple[int, int, int] | None:
    """
    Select the item with the highest estimated total contribution, using DP to estimate future value.

    Args:
        remaining_capacity: The remaining capacity of the knapsack.
        remaining_items: List of tuples containing (weight, value, index) of remaining items.

    Returns:
        The selected item as a tuple (weight, value, index), or None if no item fits.
    """
    if not remaining_items:
        return None
    
    # Filter items that fit in remaining capacity
    fitting_items = [item for item in remaining_items if item[0] <= remaining_capacity]
    
    if not fitting_items:
        return None
    
    # If there are no remaining items to fill after selection, DP is trivial, but we still need to pick the best single item.
    # However, the DP approach is useful when there are other items to consider for the "fill".
    # We build a DP table for the subset of items excluding the candidate to estimate the "fill" value.
    # But building a full DP table for each candidate is expensive O(N^2 * C).
    # A better approach for the "next step" heuristic:
    # 1. Compute a global DP table for ALL remaining items up to remaining_capacity. 
    #    Let dp[w] be the max value achievable with capacity w using ANY subset of remaining_items.
    # 2. For each fitting item i, the estimate is NOT simply dp[remaining_capacity - weight_i] because item i is already used.
    #    However, dp[remaining_capacity] is the global optimum. 
    #    If we pick item i, we want to know the max value of the rest with capacity remaining_capacity - weight_i.
    #    This is equivalent to solving the knapsack problem for remaining_items \ {i} with capacity remaining_capacity - weight_i.
    #    
    #    To avoid recomputing DP for every item, we can use the property that if the global DP solution for capacity C 
    #    includes item i, then picking i is consistent with the optimal solution structure. 
    #    But standard DP doesn't easily tell us "what if we force pick i".
    #    
    #    Alternative: Since N is likely small enough for the outer loop, and C is the capacity.
    #    We can compute the DP table once for all remaining items. 
    #    Let's compute dp[w] = max value using subset of remaining_items with capacity <= w.
    #    
    #    Actually, the previous greedy approach was O(N^2 log N) per step. 
    #    DP is O(N * C). If N is large and C is large, DP is slow. 
    #    But the prompt asks to replace the inner greedy fill with DP lookup.
    #    
    #    Let's compute the DP table for the current set of remaining_items.
    #    dp[w] = max value achievable with capacity w using items from remaining_items.
    
    # Initialize DP table
    # dp[w] will store the maximum value achievable with capacity exactly w (or up to w, let's do standard 0/1 knapsack)
    # Standard 0/1 Knapsack DP:
    # dp[j] = max value for capacity j
    
    # If remaining_capacity is large, this might be slow, but it's the requested modification.
    
    # Edge case: if remaining_items is empty after filtering, handled above.
    
    # We need to estimate the value of "fill" for each candidate.
    # The fill value for a candidate item `item` is the max value obtainable from `remaining_items \ {item}` with capacity `remaining_capacity - item.weight`.
    
    # To do this efficiently without recomputing DP for each item:
    # We can compute the forward DP and backward DP? Or just compute the global DP and assume that if an item is "critical", it might be in the optimal set.
    # However, a simpler approximation that matches the spirit of "DP lookup" is to precompute the DP table for ALL remaining items.
    # Then, for each item, we can't directly look up dp[remaining_capacity - weight] because that includes the item itself potentially.
    # 
    # Let's stick to the direct interpretation: For each fitting item, calculate the DP value of the remaining items (excluding current) for the new capacity.
    # This is O(N * N * C) which is worse than greedy if N is large.
    # 
    # Perhaps the intent is to compute the DP table once for the entire set of remaining items, and then use it to guide the choice?
    # If we compute dp[w] for all w in [0, remaining_capacity] using all remaining_items:
    # We can check if item i is part of the optimal solution for the full capacity.
    # But we need to pick ONE item now.
    # 
    # Let's try a different angle: The "future value" of picking item i is dp_excluding_i[remaining_capacity - w_i].
    # If we precompute the DP table for the whole set, we can't easily exclude i.
    # 
    # However, if we assume that the items are not too many, or the capacity is small, we can just compute the DP for the remaining items once.
    # Then, for each item i, we can attempt to reconstruct or approximate.
    # 
    # Actually, there is a known technique: 
    # If we compute the DP table `dp` for all items, we can determine if item i is used in the optimal solution for capacity `W` by checking if `dp[W] == dp[W - w_i] + v_i`.
    # But this only works if there is a unique optimal solution or if we are careful.
    # 
    # Given the constraint "replace inner greedy... with DP lookup", and the previous code did O(N) work inside the loop, 
    # computing a full DP table O(N*C) inside the loop is O(N^2*C).
    # Computing the DP table ONCE outside the loop is O(N*C).
    # Then, how to get the value excluding item i?
    # 
    # Let's compute the DP table for all remaining items.
    # dp[w] = max value with capacity w.
    # 
    # We can also compute a "backward" DP or use the property that if we pick item i, the best we can do with the rest is bounded by the global DP.
    # 
    # Let's just compute the DP table for the subset of items. 
    # To make it a "lookup", we precompute the DP table for the current `remaining_items`.
    # Then, for each item, we want the DP value of `remaining_items` without that item.
    # This is hard to get in O(1).
    # 
    # Re-reading the prompt: "Replace the inner greedy fill simulation with a dynamic programming lookup... removing the overhead of sorting and iterating... in each step."
    # The previous code sorted (N log N) and iterated (N) inside the loop. Total O(N^2 log N).
    # If we compute the DP table ONCE for all remaining items (O(N*C)), and then use it...
    # How to use it to get the value excluding item i?
    # We can't easily.
    # 
    # Maybe the idea is to use the global DP table to estimate the marginal gain?
    # Gain of item i = dp[remaining_capacity] - dp[remaining_capacity] without i? No.
    # 
    # Let's look at the standard "Knapsack with lookahead" heuristics.
    # Often, one computes the LP relaxation or the fractional knapsack value.
    # 
    # If we compute the DP table `dp` for all remaining items up to `remaining_capacity`:
    # We can define the "estimated value" of picking item i as:
    # v_i + dp_excluding_i[remaining_capacity - w_i]
    # 
    # If we cannot compute dp_excluding_i efficiently, we might approximate it by dp[remaining_capacity - w_i] IF we assume that item i is not heavily used in the optimal solution for that smaller capacity, or if w_i is small.
    # But this is inaccurate.
    # 
    # However, notice that `remaining_items` shrinks by one each step.
    # 
    # Let's try to implement the DP calculation ONCE for the current set of items.
    # Then, for each item, we can't do a perfect lookup.
    # 
    # Is there a way to do O(1) lookup for "DP without item i"?
    # Yes, if we precompute DP from left (forward) and right (backward) for sorted items? No, 0/1 knapsack doesn't have that simple prefix/suffix property because of the capacity constraint mixing.
    # 
    # Given the ambiguity, the most robust interpretation of "DP lookup" in this context, which improves over greedy sorting, is to compute the exact DP solution for the remaining capacity using ALL remaining items, and then perhaps use that to guide the selection?
    # 
    # Actually, if we compute the DP table for the remaining items, we can find the optimal subset. 
    # The item that is part of the optimal subset for the current capacity is a good candidate.
    # But we need to pick ONE.
    # 
    # Let's go with the most straightforward "DP lookup" that replaces the inner loop:
    # Precompute the DP table for `remaining_items` up to `remaining_capacity`.
    # Then, for each item i, the "fill value" is approximated by the DP value of the remaining capacity `remaining_capacity - w_i` using the REMAINING items.
    # To avoid recomputing DP, we can't.
    # 
    # Wait, if N is small, O(N * C) inside the loop is acceptable?
    # The previous code was O(N^2 log N). 
    # If C is small, DP is faster.
    # 
    # Let's implement computing the DP table for `remaining_items` ONCE.
    # Then, we can't easily exclude items.
    # 
    # Alternative: The "DP lookup" might refer to using the DP table of the ORIGINAL items (if available) but we only have remaining items.
    # 
    # Let's assume the request implies computing the DP table for the current `remaining_items` once, and then using a heuristic based on that table.
    # The most accurate "next item" heuristic using DP is to check which items are present in the optimal solution for the current capacity.
    # We can reconstruct the solution from the DP table.
    # Any item in the optimal reconstruction is a good candidate.
    # If there are multiple, pick the one with highest value/weight?
    # 
    # Let's do this:
    # 1. Compute DP table for `remaining_items` up to `remaining_capacity`.
    # 2. Reconstruct the optimal set of items for `remaining_capacity`.
    # 3. Pick the first item from the optimal set that fits (all do by definition of fit in reconstruction, but we filtered by fit already).
    # 4. If the optimal set is empty, pick None (shouldn't happen if fitting_items is not empty and we want to maximize value).
    # 
    # This effectively uses DP to select the "next" item that is part of the globally optimal solution for the current state.
    
    # Step 1: Compute DP table
    # dp[w] = max value for capacity w
    dp = [0] * (remaining_capacity + 1)
    
    # To reconstruct, we need to know which items were used.
    # We can use a boolean array or store indices.
    # Since we need to pick one item, we can just track the choices.
    # However, storing the full history is O(N*C) space.
    # Let's use a 2D DP or keep track of decisions.
    
    # Given the function signature, we just need to return one item.
    # Let's compute the DP and reconstruct the optimal set.
    
    # Optimized space for reconstruction:
    # We can compute the DP table. Then, iterate backwards to find which items were included.
    
    # Create a copy of items to process
    # Sort items? No, standard DP.
    
    # dp[w] = max value
    # keep[w] = index of item used to achieve dp[w] (or -1 if not used)
    # But multiple items can contribute. We need to know if item i was used.
    
    # Standard reconstruction:
    # Iterate items. Update DP.
    # Then backtrack.
    
    # Let's map items to indices for tracking
    # remaining_items is a list of tuples.
    
    # We'll create a DP table of size (len(remaining_items) + 1) x (remaining_capacity + 1)
    # This allows reconstruction.
    
    n = len(remaining_items)
    # dp[i][w] = max value using first i items with capacity w
    # To save memory, we can use 2 rows, but for reconstruction we need the full table or a different method.
    # Given Python, a 2D list is fine for moderate N and C.
    
    # If C is very large, this will fail. But assuming reasonable constraints for a heuristic.
    
    # Initialize DP table
    # dp[i][w]
    dp_table = [[0] * (remaining_capacity + 1) for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        w_i, v_i, idx_i = remaining_items[i-1]
        for w in range(remaining_capacity + 1):
            # Option 1: Don't take item i-1
            val_excl = dp_table[i-1][w]
            # Option 2: Take item i-1 if it fits
            val_incl = 0
            if w_i <= w:
                val_incl = dp_table[i-1][w - w_i] + v_i
            
            dp_table[i][w] = max(val_excl, val_incl)
            
    # Reconstruct the optimal set of items
    optimal_items = []
    w = remaining_capacity
    for i in range(n, 0, -1):
        if dp_table[i][w] != dp_table[i-1][w]:
            # Item i-1 was used
            optimal_items.append(remaining_items[i-1])
            w -= remaining_items[i-1][0]
            
    # optimal_items contains the items that form the optimal solution for the current remaining_capacity
    # We should pick one of these items.
    # Which one? The problem asks to "select the next item".
    # Picking any from the optimal set is a good heuristic.
    # Let's pick the one with the highest value-to-weight ratio among the optimal items to break ties?
    # Or just the first one?
    # The previous code picked the one maximizing total estimated value.
    # Here, all items in `optimal_items` contribute to the same max total value (dp_table[n][remaining_capacity]).
    # So any of them is optimal for the current state.
    # Let's pick the one with the highest value.
    
    if not optimal_items:
        # If no items are in the optimal set, it means the best strategy is to take nothing?
        # But we have fitting_items. If fitting_items is not empty, dp_table[n][remaining_capacity] > 0 unless all values are 0.
        # If values are 0, any item is fine? Or None?
        # If dp is 0, and fitting_items exist, it might mean taking an item doesn't increase value (value=0).
        # Let's return None if optimal_items is empty, implying no benefit.
        # But wait, if fitting_items is not empty, we should return one if it helps.
        # If all values are 0, best_total_value in previous code would be 0. It would return the last one checked?
        # Let's stick to the DP result. If optimal_items is empty, we can't recommend an item based on DP optimality.
        # In that case, fallback to greedy? Or return None?
        # Let's return None if optimal set is empty, as DP says taking nothing is best (or equivalent).
        return None
        
    # Select the best item from the optimal set.
    # Criteria: Highest value? Highest ratio?
    # Let's pick the one with the highest value.
    best_item = max(optimal_items, key=lambda x: x[1])
    
    return best_item

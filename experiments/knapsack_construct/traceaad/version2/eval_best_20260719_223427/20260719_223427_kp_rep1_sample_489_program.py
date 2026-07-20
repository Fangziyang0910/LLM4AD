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
    
    # Filter items that fit in the remaining capacity
    fitting_items = []
    for item in remaining_items:
        weight, value, index = item
        if weight <= remaining_capacity:
            fitting_items.append(item)
    
    if not fitting_items:
        return None
    
    if len(fitting_items) == 1:
        return fitting_items[0]
    
    # Handle zero weight items with positive value immediately
    for item in fitting_items:
        weight, value, index = item
        if weight == 0:
            if value > 0:
                return item
    
    best_item = None
    best_score = -1
    
    for candidate in fitting_items:
        c_weight, c_value, c_index = candidate
        
        # Simulate residual filling
        residual_capacity = remaining_capacity - c_weight
        
        # Get remaining items excluding the candidate
        others = [item for item in remaining_items if item[2] != c_index]
        
        # Filter others that fit in residual capacity
        fitting_others = [item for item in others if item[0] <= residual_capacity]
        
        # Sort by value density (value/weight) descending
        # Handle weight 0 carefully in sorting
        def density_key(item):
            w, v, i = item
            if w == 0:
                return float('inf')
            return v / w
        
        fitting_others_sorted = sorted(fitting_others, key=density_key, reverse=True)
        
        # 1. Standard Greedy Fill
        greedy_items = []
        current_residual_val = 0
        temp_cap = residual_capacity
        for item in fitting_others_sorted:
            w, v, i = item
            if w <= temp_cap:
                greedy_items.append(item)
                current_residual_val += v
                temp_cap -= w
                if temp_cap == 0:
                    break
        
        best_residual_val = current_residual_val
        
        # 2. Limited-Branching Lookahead: Try swapping the last added item
        # We only try swapping if there was at least one item added greedily
        if greedy_items:
            # The last item added in the greedy pass
            last_item = greedy_items[-1]
            last_w, last_v, last_i = last_item
            
            # Value of the rest of the greedy items (excluding the last one)
            rest_val = current_residual_val - last_v
            rest_weight = c_weight + (residual_capacity - temp_cap) - last_w # Total weight used before last item
            
            # Actually, let's recalculate the capacity available if we remove the last item
            # The greedy process used some capacity. Let's track used capacity explicitly.
            # But simpler: The greedy items list contains items. 
            # If we remove the last item, the capacity freed is last_w.
            # The items in greedy_items[:-1] are still taken.
            
            # Capacity used by greedy_items[:-1]
            weight_of_rest = sum(item[0] for item in greedy_items[:-1])
            capacity_for_swap = residual_capacity - weight_of_rest
            
            # Try to replace last_item with any other item from fitting_others that:
            # 1. Is not already in greedy_items[:-1]
            # 2. Fits in capacity_for_swap
            # 3. Has higher value than last_item (optional optimization, but we check max anyway)
            
            # Items already selected in the 'rest'
            rest_indices = set(item[2] for item in greedy_items[:-1])
            
            for potential_swap_item in fitting_others:
                swap_w, swap_v, swap_i = potential_swap_item
                
                # Must be a different item
                if swap_i in rest_indices:
                    continue
                    
                # Must fit in the remaining capacity after 'rest' items
                if swap_w <= capacity_for_swap:
                    # Calculate total value if we pick 'rest' + potential_swap_item
                    swap_total_val = rest_val + swap_v
                    if swap_total_val > best_residual_val:
                        best_residual_val = swap_total_val

        total_score = c_value + best_residual_val
        
        if total_score > best_score:
            best_score = total_score
            best_item = candidate
        elif total_score == best_score:
            # Tie-breaker: prefer higher value
            if c_value > best_item[1]:
                best_item = candidate
            elif c_value == best_item[1]:
                # Further tie-breaker: prefer lighter weight (leaves more room)
                if c_weight < best_item[0]:
                    best_item = candidate
    
    return best_item

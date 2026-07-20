import math
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
    fitting_items = [(w, v, i) for w, v, i in remaining_items if w <= remaining_capacity]
    
    if not fitting_items:
        return None
        
    # If only one fits, return it
    if len(fitting_items) == 1:
        return fitting_items[0]
    
    # Precompute items sorted by value/weight for greedy fill estimation
    # Sort all remaining items by density (value/weight) descending
    # Handle zero weight case by assigning infinite density
    def get_density(item):
        w, v, i = item
        if w == 0:
            return float('inf') if v > 0 else 0
        return v / w

    all_items_sorted = sorted(remaining_items, key=get_density, reverse=True)
    
    best_item = None
    best_total_value = -1
    best_stability = -1
    
    # Pre-calculate stability for fitting items to avoid recomputation if possible, 
    # but since it depends on remaining_capacity which is constant for this call, 
    # we can compute it on demand or cache. 
    # Given N is likely small enough for O(N log N) sorting, O(N^2) logic was the bottleneck.
    # Now we do O(N) per candidate, so total O(N^2) or O(N log N * N) depending on sort reuse.
    # Since we sort all_items_sorted once, and iterate fitting_items, it's roughly O(N^2) worst case
    # but with a much smaller constant factor than the previous O(N^2) swap search which was nested inside.
    
    for candidate in fitting_items:
        cw, cv, ci = candidate
        remaining_cap_after = remaining_capacity - cw
        
        # Step 1: Greedy fill with remaining items (excluding candidate)
        selected_set = [] # List of items selected in greedy phase
        unselected_set = [] # List of items not selected in greedy phase
        current_cap = remaining_cap_after
        
        # We need to distinguish between the specific instance of candidate and others if duplicates exist
        # Using index for uniqueness
        
        temp_selected = []
        temp_unselected = []
        
        # Iterate through sorted items to build greedy sets
        for item in all_items_sorted:
            # Skip the candidate itself
            if item[2] == ci and item[0] == cw and item[1] == cv:
                continue
            
            iw, iv, ii = item
            if iw <= current_cap:
                temp_selected.append(item)
                current_cap -= iw
            else:
                temp_unselected.append(item)
                
        selected_set = temp_selected
        unselected_set = temp_unselected
        
        # Calculate base greedy value
        base_value = cv + sum(iv for iw, iv, ii in selected_set)
        current_total_value = base_value
        
        # Step 2: Linear "best-fit swap" check
        # Identify the item in selected_set with the lowest value-to-weight ratio
        # And attempt to replace it with the highest value-to-weight ratio item from unselected_set
        # that fits the remaining capacity after removal.
        
        if selected_set:
            # Find item with lowest density in selected_set
            # We want to remove the "least efficient" item to make room for a "more efficient" one
            
            # Get the item with min density
            # Note: If multiple items have same min density, any one is fine, 
            # but we should pick the one that allows the best swap. 
            # To keep it strictly linear O(N), we pick the single lowest density item.
            
            min_density_item = None
            min_density = float('inf')
            
            for item in selected_set:
                w, v, i = item
                d = v/w if w > 0 else 0 # Zero weight items have high density usually, unless value 0
                # Actually, if w=0, density is inf. We probably don't want to remove those.
                # Let's treat w=0 as very high density so they stay.
                if w == 0:
                    d = float('inf')
                else:
                    d = v / w
                
                if d < min_density:
                    min_density = d
                    min_density_item = item
            
            # If we found a candidate to remove, try to swap with best from unselected
            if min_density_item:
                rw, rv, ri = min_density_item
                
                # Capacity available after removing this item
                # The selected_set currently occupies some weight.
                # Let's calculate the weight of selected_set
                selected_weight = sum(iw for iw, iv, ii in selected_set)
                
                # Weight after removing min_density_item
                weight_after_remove = selected_weight - rw
                
                # Available capacity for the new item
                available_capacity = remaining_cap_after - weight_after_remove
                
                # Find best item in unselected_set that fits in available_capacity
                # "Best" is defined by highest value-to-weight ratio (density)
                
                best_swap_item = None
                best_swap_density = -1
                
                for item in unselected_set:
                    uw, uv, ui = item
                    
                    if uw <= available_capacity:
                        # Calculate density
                        if uw == 0:
                            d = float('inf') if uv > 0 else 0
                        else:
                            d = uv / uw
                        
                        if d > best_swap_density:
                            best_swap_density = d
                            best_swap_item = item
                
                # If we found a valid swap, calculate new total value
                if best_swap_item:
                    _, new_v, _ = best_swap_item
                    # New value = Base Value - rv + new_v
                    new_total_value = base_value - rv + new_v
                    
                    if new_total_value > current_total_value:
                        current_total_value = new_total_value

        total_value = current_total_value
        
        # Compare with best found so far
        if total_value > best_total_value:
            best_total_value = total_value
            best_item = candidate
            # Calculate stability for tie-breaking later if needed, or just store it
            if cw > 0:
                best_stability = cv / math.sqrt(cw * remaining_capacity)
            else:
                best_stability = float('inf') if cv > 0 else -1
        elif total_value == best_total_value and best_item is not None:
            # Tie-breaking: use the updated stability metric
            if cw > 0:
                cand_stability = cv / math.sqrt(cw * remaining_capacity)
            else:
                cand_stability = float('inf') if cv > 0 else -1
            
            if cand_stability > best_stability:
                best_item = candidate
                best_stability = cand_stability

    return best_item

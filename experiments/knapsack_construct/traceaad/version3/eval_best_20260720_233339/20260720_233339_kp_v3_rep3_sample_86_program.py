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
    if not remaining_items:
        return None

    # Filter items that fit in the remaining capacity
    fitting_items = [item for item in remaining_items if item[0] <= remaining_capacity]
    
    if not fitting_items:
        return None

    def get_density(item: Tuple[int, int, int]) -> float:
        w, v, idx = item
        if w == 0:
            return float('inf')
        return v / w

    def get_fractional_upper_bound(capacity: int, items: List[Tuple[int, int, int]]) -> float:
        """
        Calculates the Fractional Knapsack Upper Bound for a given capacity and set of items.
        """
        if capacity <= 0 or not items:
            return 0.0
        
        # Sort by density descending
        sorted_items = sorted(items, key=get_density, reverse=True)
        
        total_value = 0.0
        current_cap = capacity
        
        for item in sorted_items:
            w, v, idx = item
            if w == 0:
                total_value += v
                continue
            
            if current_cap <= 0:
                break
                
            if w <= current_cap:
                total_value += v
                current_cap -= w
            else:
                fraction = current_cap / w
                total_value += fraction * v
                current_cap = 0
                break
                
        return total_value

    def get_local_search_greedy_value(capacity: int, items: List[Tuple[int, int, int]], top_k: int = 10) -> float:
        """
        Estimates the achievable integer value using a greedy fill with local swap perturbation.
        1. Construct an initial greedy pack using top-K densest items that fit.
        2. Iteratively try to improve the pack by removing an item and adding a better one.
        """
        if capacity <= 0 or not items:
            return 0.0
            
        # Sort all available items by density descending for the initial selection and potential replacements
        sorted_items = sorted(items, key=get_density, reverse=True)
        
        # Step 1: Initial Greedy Pack
        current_pack = [] # List of items currently in the knapsack
        current_used_indices = set()
        current_weight = 0
        current_value = 0
        
        for item in sorted_items:
            if len(current_pack) >= top_k:
                # We limit the initial pack size to top_k to keep it manageable, 
                # though technically greedy fills as much as possible. 
                # However, for "bounded" greedy, we just take top K candidates that fit.
                # Let's stick to the previous logic: take top K densest that fit greedily.
                pass
            
            w, v, idx = item
            if w <= (capacity - current_weight):
                current_pack.append(item)
                current_used_indices.add(idx)
                current_weight += w
                current_value += v
        
        # Step 2: Local Swap Perturbation
        # Try to improve the pack by swapping out items.
        # Strategy: Iterate through items in the current pack. Try removing one.
        # If removed, try to add the best fitting item from the unused set.
        # If the new value is higher, keep the change.
        
        # To maximize improvement, we should consider removing items that are "low value" or "blocking" better items.
        # A simple heuristic: try removing each item in the pack one by one (starting from the one with lowest value/density?)
        # and see if we can fit a better combination.
        
        # Let's iterate a fixed number of swap attempts to keep complexity low.
        improved = True
        iterations = 0
        max_iterations = 10 # Limit local search steps
        
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            
            # Create a list of items in the current pack to evaluate for removal
            # Sort by value ascending (remove least valuable first) or density ascending?
            # Removing low value items frees weight for potentially high value items.
            pack_sorted_by_value_asc = sorted(current_pack, key=lambda x: x[1])
            
            for i, item_to_remove in enumerate(pack_sorted_by_value_asc):
                rem_w, rem_v, rem_idx = item_to_remove
                
                # Temporarily remove
                new_weight = current_weight - rem_w
                new_value = current_value - rem_v
                temp_unused_indices = current_used_indices - {rem_idx}
                
                # Find the best item from the unused set that fits in the remaining space
                best_new_item = None
                best_new_val = -1
                
                # Search in descending density order to find the best fit
                for cand in sorted_items:
                    c_w, c_v, c_idx = cand
                    if c_idx not in temp_unused_indices:
                        if c_w <= (capacity - new_weight):
                            # Calculate potential new value if we add this
                            pot_val = new_value + c_v
                            if pot_val > best_new_val:
                                best_new_val = pot_val
                                best_new_item = cand
                                # We can break here if we want the first best density, 
                                # but since we want max value, and density correlates, 
                                # checking all might be safer, but expensive. 
                                # Given sorted by density, the first one that fits might not be max value if weights vary wildly.
                                # But for a quick local search, picking the first high-density one that fits is a decent heuristic.
                                # Let's just pick the one with max value that fits.
                                # To do this efficiently, we'd need to check all. 
                                # Let's optimize: only check top K unused items.
                                pass
                
                # Actually, let's just find the max value item that fits.
                max_val_candidate = None
                max_val_candidate_val = -1
                
                for cand in sorted_items:
                    c_w, c_v, c_idx = cand
                    if c_idx not in temp_unused_indices:
                        if c_w <= (capacity - new_weight):
                            if c_v > max_val_candidate_val:
                                max_val_candidate_val = c_v
                                max_val_candidate = cand
                
                if max_val_candidate:
                    new_potential_value = new_value + max_val_candidate_val
                    if new_potential_value > current_value:
                        # Perform the swap
                        current_pack.remove(item_to_remove)
                        current_used_indices.remove(rem_idx)
                        
                        current_pack.append(max_val_candidate)
                        current_used_indices.add(max_val_candidate[2])
                        
                        current_weight = new_weight + max_val_candidate[0]
                        current_value = new_potential_value
                        
                        improved = True
                        break # Restart evaluation after change
        
        return current_value

    def evaluate_residual_hybrid_score(capacity: int, items: List[Tuple[int, int, int]], top_k: int = 10) -> float:
        """
        Calculates a hybrid score for the residual capacity using Local Search Greedy as the base estimate.
        """
        if capacity <= 0 or not items:
            return 0.0
            
        # Use the local search greedy value as the estimate for achievable residual value
        greedy_value = get_local_search_greedy_value(capacity, items, top_k)
        
        return greedy_value

    best_item = None
    best_score = -1.0
    best_value = -1
    best_index = float('inf')

    # Set K for bounded greedy fill / local search
    K = 10

    for item in fitting_items:
        w, v, idx = item
        
        remaining_cap_after = remaining_capacity - w
        
        # Get other items
        other_items = [i for i in remaining_items if i[2] != idx]
        
        # Evaluate the residual capacity's potential using the hybrid metric
        residual_value_estimate = evaluate_residual_hybrid_score(remaining_cap_after, other_items, K)
        
        # Total Score = Immediate Value + Estimated Residual Value
        score = v + residual_value_estimate
        
        # Tie-breaking: highest score, then highest value, then lowest index
        if best_item is None:
            best_item = item
            best_score = score
            best_value = v
            best_index = idx
        else:
            if score > best_score:
                best_item = item
                best_score = score
                best_value = v
                best_index = idx
            elif score == best_score:
                if v > best_value:
                    best_item = item
                    best_score = score
                    best_value = v
                    best_index = idx
                elif v == best_value:
                    if idx < best_index:
                        best_item = item
                        best_score = score
                        best_value = v
                        best_index = idx
                        
    return best_item

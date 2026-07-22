from typing import List, Tuple
import math

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
    
    # Filter items that fit
    fitable_items = []
    for item in remaining_items:
        w, v, idx = item
        if w <= remaining_capacity:
            fitable_items.append(item)
            
    if not fitable_items:
        return None
    
    # If only one item fits, return it
    if len(fitable_items) == 1:
        return fitable_items[0]
    
    best_item = None
    best_score = -float('inf')
    best_item_own_value = -float('inf')
    best_item_idx = float('inf')
    
    # Helper to calculate integer greedy residual value and residual density gradient penalty for a set of items and capacity
    def get_score_with_penalty(items: List[Tuple[int, int, int]], capacity: int) -> float:
        if capacity <= 0 or not items:
            return 0.0
        
        # Filter out items with weight 0 separately as they fit always and have infinite density
        zero_weight_value = 0.0
        valid_items = []
        for item in items:
            w, v, idx = item
            if w == 0:
                if v > 0:
                    zero_weight_value += v
            else:
                valid_items.append(item)
        
        if not valid_items:
            # Only zero weight items remain
            return zero_weight_value

        # Sort valid items by density descending
        sorted_items = sorted(valid_items, key=lambda x: x[1]/x[0], reverse=True)
        
        current_cap = capacity
        total_value = zero_weight_value
        selected_items_indices = set() # Track which items were selected in the greedy step
        
        for w, v, idx in sorted_items:
            if current_cap < w:
                # Cannot fit this whole item
                continue
            
            total_value += v
            current_cap -= w
            selected_items_indices.add(idx)
            
            if current_cap <= 0:
                break
        
        # Calculate Residual Density Gradient Penalty
        penalty = 0.0
        
        # Determine the set of remaining items after greedy selection
        # These are items not in selected_items_indices
        remaining_after_greedy = [it for it in valid_items if it[2] not in selected_items_indices]
        
        # Identify items that fit in the leftover capacity
        items_fitting_in_leftover = [it for it in remaining_after_greedy if it[0] <= current_cap]
        
        if current_cap > 0:
            if items_fitting_in_leftover:
                # Calculate average density of items that fit in the leftover capacity
                total_density = 0.0
                for w, v, idx in items_fitting_in_leftover:
                    total_density += v / w
                residual_avg_density = total_density / len(items_fitting_in_leftover)
            else:
                # No items fit in the leftover capacity, density is 0
                residual_avg_density = 0.0
                
            # The penalty logic requires the candidate's density. 
            # However, this helper function is called for the *other* items. 
            # The "candidate" is the item being evaluated in the outer loop.
            # But wait, the prompt says: "subtract a penalty proportional to the difference between the candidate's density and this residual average".
            # This implies the penalty calculation must happen in the context of the candidate, not inside the generic helper for residual items.
            # The current helper structure `get_score_with_penalty` only knows about `items` and `capacity`. 
            # It does not know which item was the "candidate" that led to this state.
            # Therefore, we cannot calculate the penalty *inside* this helper if it depends on the specific candidate's density.
            # We must restructure or pass candidate density.
            
            # Let's change the approach: The helper returns the raw greedy value and the leftover capacity and residual avg density info.
            # But the function signature is fixed to return float.
            # We can't easily change the signature.
            # However, the prompt asks to replace the penalty logic.
            # The previous penalty was calculated inside `get_score_with_penalty` using `max_density_of_unselected_items`.
            # That density belonged to the *residual* items.
            # The new penalty depends on *candidate* density AND *residual* density.
            # Since `get_score_with_penalty` is called for `other_items` with `residual_cap`, it doesn't know the candidate.
            # We must calculate the greedy part inside the loop, then calculate the penalty in the loop.
            
            return total_value # Return just the greedy value, we will handle penalty outside

        return total_value

    for item in fitable_items:
        w, v, idx = item
        
        # Remaining capacity after selecting this item
        residual_cap = remaining_capacity - w
        
        # Remaining items excluding the current candidate
        other_items = [it for it in remaining_items if it[2] != idx]
        
        # Calculate Candidate Density
        if w == 0:
            candidate_density = float('inf')
        else:
            candidate_density = v / w
            
        # Perform Integer Greedy Look-Ahead for other items in residual_cap
        # We need to replicate the greedy logic here to get the leftover capacity and residual avg density
        # because the helper doesn't return that info and penalty depends on candidate density.
        
        if residual_cap <= 0 or not other_items:
            greedy_value = 0.0
            leftover_cap = residual_cap
            residual_avg_density = 0.0
        else:
            # Filter out items with weight 0
            zero_weight_value = 0.0
            valid_other_items = []
            for other_item in other_items:
                ow, ov, oidx = other_item
                if ow == 0:
                    if ov > 0:
                        zero_weight_value += ov
                else:
                    valid_other_items.append(other_item)
            
            if not valid_other_items:
                greedy_value = zero_weight_value
                leftover_cap = residual_cap
                residual_avg_density = 0.0
            else:
                # Sort valid items by density descending
                sorted_other_items = sorted(valid_other_items, key=lambda x: x[1]/x[0], reverse=True)
                
                current_cap = residual_cap
                greedy_value = zero_weight_value
                selected_other_indices = set()
                
                for ow, ov, oidx in sorted_other_items:
                    if current_cap < ow:
                        continue
                    greedy_value += ov
                    current_cap -= ow
                    selected_other_indices.add(oidx)
                    if current_cap <= 0:
                        break
                
                leftover_cap = current_cap
                
                # Determine remaining items after greedy selection
                remaining_after_greedy = [it for it in valid_other_items if it[2] not in selected_other_indices]
                
                # Identify items that fit in the leftover capacity
                items_fitting_in_leftover = [it for it in remaining_after_greedy if it[0] <= leftover_cap]
                
                if leftover_cap > 0 and items_fitting_in_leftover:
                    total_density = 0.0
                    for ow, ov, oidx in items_fitting_in_leftover:
                        total_density += ov / ow
                    residual_avg_density = total_density / len(items_fitting_in_leftover)
                else:
                    residual_avg_density = 0.0

        # Calculate Residual Density Gradient Penalty
        # Penalty = (Candidate_Density - Residual_Avg_Density) * Leftover_Capacity * (Leftover_Capacity / Remaining_Capacity)
        # If Candidate_Density is inf, we handle it. If w=0, it fits perfectly with no leftover usually unless capacity > 0 and no other items.
        # If w=0, candidate_density is inf. Diff is inf. Penalty is inf?
        # If w=0, it consumes 0 capacity. Leftover = residual_cap.
        # If we pick a 0-weight item, we should always pick it if value > 0?
        # The penalty logic might break for 0-weight items.
        # If w=0, candidate_density is inf. 
        # If residual_avg_density is finite, diff is inf. Penalty is inf. Score becomes -inf.
        # This would penalize picking 0-weight items, which is wrong.
        # 0-weight items are free value. They should always be picked.
        # Let's treat 0-weight items specially: if w==0, penalty is 0.
        
        penalty = 0.0
        if w > 0:
            if residual_cap > 0: # Only calculate if there is leftover
                 # If residual_avg_density is 0, it means no items fit in leftover.
                 diff = candidate_density - residual_avg_density
                 if diff > 0:
                     # Scale factor: ratio of leftover to total remaining capacity
                     ratio = leftover_cap / remaining_capacity
                     penalty = diff * leftover_cap * ratio
            # else: leftover_cap is 0, penalty is 0
            
        score = v + greedy_value - penalty
        
        # Tie-breaking: higher score, then higher value, then lower index
        if score > best_score + 1e-9:
            best_item = item
            best_score = score
            best_item_own_value = v
            best_item_idx = idx
        elif abs(score - best_score) <= 1e-9:
            # Tie in score
            if v > best_item_own_value:
                best_item = item
                best_score = score
                best_item_own_value = v
                best_item_idx = idx
            elif v == best_item_own_value:
                if idx < best_item_idx:
                    best_item = item
                    best_score = score
                    best_item_own_value = v
                    best_item_idx = idx
                        
    return best_item

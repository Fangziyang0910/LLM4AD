
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
    import math
    from typing import List, Tuple

    if not remaining_items:
        return None

    # Filter feasible items
    feasible_items = [item for item in remaining_items if item[0] <= remaining_capacity]
    
    if not feasible_items:
        return None

    # Handle zero weight positive value items as they are always optimal to pick
    zero_weight_positive = [item for item in feasible_items if item[0] == 0 and item[1] > 0]
    if zero_weight_positive:
        # Among zero weight positive, pick the one with highest value. 
        # If tied, pick lowest index.
        best_zero = max(zero_weight_positive, key=lambda x: (x[1], -x[2]))
        return best_zero

    # If no zero-weight positive items, proceed with heuristic search for positive weight items
    if not feasible_items:
        return None

    # Design Idea: Capacity-Partitioned Multi-Score Lookahead
    # We simulate the greedy fill for each feasible candidate as the "next" item.
    # However, instead of a single efficiency score, we use a composite score derived from 
    # three distinct perspectives of the remaining items' sort order:
    # 1. Standard Value Density (v/w)
    # 2. Weight Compactness (1/w, favoring small items to allow more combinations)
    # 3. Fit Quality (1 - (cap % w)/cap, favoring items that leave little waste)
    
    # We define a simulation function that takes a specific scoring key and runs the greedy fill.
    
    def simulate_greedy_fill(capacity: int, items: List[Tuple[int, int, int]], sort_key_func) -> int:
        """
        Simulates filling the knapsack with the given items sorted by sort_key_func.
        Returns the total value achieved.
        """
        sorted_items = sorted(items, key=sort_key_func, reverse=True)
        current_cap = capacity
        total_value = 0
        
        for item in sorted_items:
            w, v, idx = item
            if w <= current_cap:
                current_cap -= w
                total_value += v
                
        return total_value

    # Define the three heuristic sort keys
    def score_density(item: Tuple[int, int, int]) -> float:
        """Standard Value to Weight Ratio"""
        w, v, idx = item
        if w == 0:
            return float('inf') if v > 0 else 0
        return v / w

    def score_compactness(item: Tuple[int, int, int]) -> float:
        """Inverse Weight (prefers smaller items)"""
        w, v, idx = item
        if w == 0:
            return float('inf')
        return 1.0 / w

    def score_fit_quality(item: Tuple[int, int, int]) -> float:
        """Fit Quality relative to current capacity (prefers items that divide capacity evenly)"""
        w, v, idx = item
        if w == 0:
            return float('inf') if v > 0 else 0
        # We need to capture the capacity context for this score, 
        # but sort_key_func usually only takes item. 
        # To make this work in a general sort, we can't easily inject 'capacity' into a simple sort key 
        # without partial application or a wrapper. 
        # For the simulation, we are sorting the *remaining* candidates. 
        # Let's use a partial function approach or a wrapper class if necessary. 
        # For simplicity in this signature, let's approximate fit quality by:
        # Value / (Weight + (Weight % Capacity)?). No, that's complex.
        # Let's stick to: v / w * (1 - (capacity % w)/capacity)
        # But since 'capacity' changes during fill, this is hard to pre-calculate for the whole list.
        # Alternative: Use a "Lookahead Fit" score calculated at the time of sorting.
        # However, the items list is static. The capacity changes.
        # A static sort key cannot accurately reflect dynamic capacity fit.
        
        # Correction: The previous algorithm calculated score *inside* the loop or based on current_cap.
        # Here, to keep the sort simple for the simulation, we will use a static heuristic 
        # that approximates fit: items with weights that are factors of typical capacities?
        # Or simply: Value Density is the most robust static metric.
        # Let's introduce a 3rd metric: "Value per unit of 'Waste Potential'"?
        # Let's try: v / (w * (1 + 0.1 * (w/100))) -> Penalize large items slightly more than density?
        # Let's stick to the prompt's request for "novel mechanisms".
        
        # Let's use a dynamic scoring inside the simulation loop instead of a static sort key?
        # That would be O(N log N) per simulation, which is fine.
        return v / w # Fallback to density if we can't do dynamic

    # To truly implement "Fit Quality" dynamically, we modify the simulate function.
    
    def simulate_dynamic_greedy(capacity: int, items: List[Tuple[int, int, int]], strategy: str) -> int:
        """
        Simulates filling using a specific strategy.
        Strategy 'density': sort by v/w.
        Strategy 'compactness': sort by 1/w.
        Strategy 'fit': sort dynamically by (v/w) * (1 - (current_cap % w)/current_cap).
        """
        # Make a copy to avoid modifying original list order if it matters elsewhere
        candidates = list(items)
        
        if strategy == 'density':
            # Pre-sort by density
            candidates.sort(key=lambda i: i[1]/i[0] if i[0]>0 else float('inf'), reverse=True)
        elif strategy == 'compactness':
            # Pre-sort by inverse weight
            candidates.sort(key=lambda i: 1.0/i[0] if i[0]>0 else float('inf'), reverse=True)
        elif strategy == 'fit':
            # Dynamic sort: We can't pre-sort because the score depends on current_cap.
            # We must select the best item at each step. This is O(N^2) per simulation.
            # Given N is likely small enough for this lookahead, this is acceptable.
            pass 
            
        current_cap = capacity
        total_value = 0
        
        if strategy == 'fit':
            # Re-evaluate best item at each step based on dynamic fit
            while current_cap > 0 and candidates:
                best_item = None
                best_score = -float('inf')
                
                # Find the item with the highest dynamic fit score
                # Score = (v/w) * (1 - (current_cap % w)/current_cap)
                # Note: If w > current_cap, it's not feasible.
                
                # To optimize, we can filter feasible first
                feasible_now = [i for i in candidates if i[0] <= current_cap]
                if not feasible_now:
                    break
                
                for item in feasible_now:
                    w, v, idx = item
                    if w == 0:
                        score = float('inf') if v > 0 else 0
                    else:
                        remainder = current_cap % w
                        fit_factor = 1.0 - (remainder / current_cap)
                        density = v / w
                        score = density * fit_factor
                    
                    if score > best_score:
                        best_score = score
                        best_item = item
                    elif score == best_score:
                        # Tie-breaker for dynamic selection: highest value, then lowest index
                        if best_item is None:
                            best_item = item
                        else:
                            if item[1] > best_item[1]:
                                best_item = item
                            elif item[1] == best_item[1] and item[2] < best_item[2]:
                                best_item = item

                if best_item:
                    w, v, idx = best_item
                    total_value += v
                    current_cap -= w
                    candidates.remove(best_item) # Remove selected item
                else:
                    break
        else:
            # Static sort strategies
            for item in candidates:
                w, v, idx = item
                if w <= current_cap:
                    current_cap -= w
                    total_value += v
                # Note: In static sort, we just iterate. If an item doesn't fit, we skip it.
                # The list remains in the same order.

        return total_value

    best_projection_total_value = -float('inf')
    best_starting_item = None
    
    # We will test each feasible item as the starting item.
    # For each starting item, we simulate the rest of the fill using the 3 strategies.
    # We take the MAX value across all strategies for that starting item as its "Projected Value".
    
    for start_item in feasible_items:
        w_start, v_start, idx_start = start_item
        
        current_cap = remaining_capacity - w_start
        current_total_value = v_start
        
        # Remaining items excluding the start item
        # We must ensure we don't pick the start item again.
        # Using 'is' or identity check might be risky if duplicates exist in value/weight/index.
        # Since items are tuples, we assume unique identity or handle duplicates by removing one instance.
        # To be safe, we create a list of remaining items that are not the start_item.
        # Assuming items in the list are distinct objects or we match by all 3 fields.
        remaining_candidates = [item for item in feasible_items if item != start_item]
        
        # Simulate with 3 different strategies
        val_density = simulate_dynamic_greedy(current_cap, remaining_candidates, 'density')
        val_compact = simulate_dynamic_greedy(current_cap, remaining_candidates, 'compactness')
        val_fit = simulate_dynamic_greedy(current_cap, remaining_candidates, 'fit')
        
        # The projected total value for this start item is the start value + max of the rest
        max_rest_value = max(val_density, val_compact, val_fit)
        total_projected = v_start + max_rest_value
        
        # Update best if this simulation is better
        if total_projected > best_projection_total_value:
            best_projection_total_value = total_projected
            best_starting_item = start_item
        elif total_projected == best_projection_total_value:
            # Tie-breaker: 
            # 1. Prefer the item with higher value itself
            # 2. If tied, prefer lower weight (to save capacity for future steps? or more flexibility?)
            # 3. If tied, prefer lower index
            if best_starting_item is None:
                best_starting_item = start_item
            else:
                if start_item[1] > best_starting_item[1]:
                    best_starting_item = start_item
                elif start_item[1] == best_starting_item[1]:
                    if start_item[0] < best_starting_item[0]:
                        best_starting_item = start_item
                    elif start_item[0] == best_starting_item[0]:
                        if start_item[2] < best_starting_item[2]:
                            best_starting_item = start_item

    if best_starting_item:
        return best_starting_item
    
    # Fallback: if simulations didn't yield a result (e.g. all items negative value), 
    # pick the one with highest value that fits (standard greedy)
    feasible_items.sort(key=lambda x: x[1], reverse=True)
    if feasible_items:
        return feasible_items[0]

    return None

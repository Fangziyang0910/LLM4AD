from typing import List, Tuple
import statistics

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
    feasible_items = [item for item in remaining_items if item[0] <= remaining_capacity]
    
    if not feasible_items:
        return None

    # Helper function for recursive greedy potential estimation
    def get_recursive_potential(capacity: int, items: List[Tuple[int, int, int]], max_depth: int, current_depth: int) -> float:
        """
        Estimates the maximum value achievable from 'items' in 'capacity' using a greedy approach
        with limited recursion depth.
        
        Args:
            capacity: The knapsack capacity available.
            items: List of available items (weight, value, index).
            max_depth: The maximum allowed recursion depth.
            current_depth: The current recursion depth.
            
        Returns:
            Estimated total value.
        """
        # Base case: depth limit reached or no capacity/items
        if capacity <= 0 or not items or current_depth >= max_depth:
            # Use a simple greedy estimate (0/1) for the remaining capacity
            sorted_items = sorted(items, key=lambda x: (x[1] / max(x[0], 1e-9), x[1]), reverse=True)
            est_val = 0.0
            curr_cap = capacity
            for w, v, _ in sorted_items:
                if w <= curr_cap:
                    est_val += v
                    curr_cap -= w
            return est_val

        # Determine feasible items for this step
        next_feasible = [item for item in items if item[0] <= capacity]
        if not next_feasible:
            return 0.0

        # Sort by value-to-weight ratio to guide greedy choice
        next_feasible.sort(key=lambda x: (x[1] / max(x[0], 1e-9), x[1]), reverse=True)
        
        # Limit branching to top K items to keep performance reasonable
        # K=3 is a reasonable trade-off between accuracy and speed
        candidates_to_try = next_feasible[:3]
        
        best_estimated_total = 0.0
        
        for item in candidates_to_try:
            w, v, idx = item
            # Remove current item from list for next recursion level
            # Using index for unique identification
            remaining_after = [i for i in items if i[2] != idx]
            
            # Calculate potential value: current item value + estimated future value
            potential = v + get_recursive_potential(capacity - w, remaining_after, max_depth, current_depth + 1)
            if potential > best_estimated_total:
                best_estimated_total = potential
                
        return best_estimated_total

    # Calculate dynamic max depth based on "residual fragmentation risk"
    # Metric: Ratio of remaining_capacity to median weight of feasible items
    # If ratio is low, capacity is tight relative to item sizes -> higher risk of fragmentation -> deeper search
    # If ratio is high, capacity is abundant -> lower risk -> shallow search
    
    if len(feasible_items) > 0:
        weights = [item[0] for item in feasible_items]
        median_weight = statistics.median(weights)
        
        # Avoid division by zero if median weight is 0 (unlikely for valid items, but safe to handle)
        if median_weight > 0:
            frag_ratio = remaining_capacity / median_weight
            
            # Define thresholds for depth mapping
            # Low frag_ratio (< 2): High risk, depth 6
            # High frag_ratio (> 10): Low risk, depth 1
            # Interpolate between 2 and 10
            
            min_ratio = 2.0
            max_ratio = 10.0
            min_depth = 6
            max_depth = 1
            
            if frag_ratio <= min_ratio:
                dynamic_depth = min_depth
            elif frag_ratio >= max_ratio:
                dynamic_depth = max_depth
            else:
                # Linear interpolation
                # depth decreases as ratio increases
                fraction = (frag_ratio - min_ratio) / (max_ratio - min_ratio)
                dynamic_depth = min_depth - fraction * (min_depth - max_depth)
                dynamic_depth = int(round(dynamic_depth))
                # Clamp depth between 1 and 6
                dynamic_depth = max(1, min(6, dynamic_depth))
        else:
            # If median weight is 0, items are free? Just pick greedy/shallow
            dynamic_depth = 1
    else:
        dynamic_depth = 1

    def calculate_selection_key(item: Tuple[int, int, int]) -> Tuple[float, float, int]:
        weight, value, index = item
        
        # Exclude current item from remaining for the simulation
        remaining_after = [i for i in remaining_items if i[2] != index]
        
        residual_cap = remaining_capacity - weight
        
        # Estimate future value using dynamic depth derived from fragmentation risk
        future_value_est = get_recursive_potential(residual_cap, remaining_after, max_depth=dynamic_depth, current_depth=0)
        total_potential = value + future_value_est
        
        # Value-to-weight ratio (tie-breaker)
        if weight > 0:
            ratio = value / weight
        else:
            ratio = 0.0
            
        # Tuple: (total_potential, ratio, value)
        return (total_potential, ratio, value)

    # Select the item with the best key
    best_item = max(feasible_items, key=calculate_selection_key)
    
    return best_item

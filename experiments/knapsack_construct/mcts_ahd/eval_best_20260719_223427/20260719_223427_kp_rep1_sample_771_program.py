
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
    if remaining_capacity <= 0:
        return None

    # Filter feasible items
    feasible_items = [item for item in remaining_items if item[0] <= remaining_capacity]
    
    if not feasible_items:
        return None

    # Pre-calculate the weight of the smallest remaining item for fragmentation check
    remaining_weights = [w for w, v, idx in remaining_items]
    min_weight = min(remaining_weights) if remaining_weights else float('inf')

    # Sort all remaining items by value density descending for greedy look-ahead
    # Handle division by zero gracefully
    sorted_items = sorted(
        remaining_items, 
        key=lambda x: x[1]/x[0] if x[0] > 0 else float('inf'), 
        reverse=True
    )

    best_item = None
    best_score = -float('inf')

    # Total weight of all remaining items for normalization
    total_remaining_weight = sum(w for w, v, idx in remaining_items)
    total_remaining_value = sum(v for w, v, idx in remaining_items)

    for weight, value, index in feasible_items:
        immediate_value = value
        residual_capacity = remaining_capacity - weight
        
        # Look-ahead: Greedily estimate the value of packing the rest of the items
        projected_future_value = 0
        temp_capacity = residual_capacity
        packed_weight_future = 0
        items_considered = 0
        
        for w, v, idx in sorted_items:
            if idx == index: # Skip current item as it's already picked
                continue
            if w <= temp_capacity:
                projected_future_value += v
                temp_capacity -= w
                packed_weight_future += w
                items_considered += 1
        
        # Calculate Utilization Factor
        # Rewards high utilization of residual capacity
        if residual_capacity == 0:
            utilization_factor = 1.0
        elif packed_weight_future == 0:
            # If no future items fit, penalize based on wasted capacity
            utilization_factor = max(0.0, 1.0 - (residual_capacity / (remaining_capacity + 1e-9)))
        else:
            utilization_ratio = packed_weight_future / (residual_capacity + 1e-9)
            # New mechanism: Exponential decay for low utilization to punish poor fills more harshly
            utilization_factor = 0.6 + 0.4 * (utilization_ratio ** 1.5)

        # Fragmentation Penalty
        # Penalize if the residual capacity is non-zero but smaller than the smallest remaining item
        fragmentation_penalty = 0.0
        if residual_capacity > 0 and residual_capacity < min_weight:
            # New mechanism: Penalty scales with the square of the gap ratio to heavily discourage tiny gaps
            gap_ratio = residual_capacity / (remaining_capacity + 1e-9)
            fragmentation_penalty = (gap_ratio ** 2) * immediate_value * 1.2

        # Density Bonus
        # Reward items with high value density relative to the average density of remaining items
        avg_density = total_remaining_value / (total_remaining_weight + 1e-9)
        item_density = value / (weight + 1e-9)
        density_bonus = 0
        if item_density > avg_density:
            # Bonus for high density items, scaled by how much better they are than average
            density_bonus = (item_density / avg_density - 1) * immediate_value * 0.1

        # Final Score Calculation
        # Combine immediate value, projected future value (scaled by utilization), fragmentation penalty, and density bonus
        total_score = immediate_value + (projected_future_value * utilization_factor) - fragmentation_penalty + density_bonus
        
        if total_score > best_score:
            best_score = total_score
            best_item = (weight, value, index)
            
    return best_item

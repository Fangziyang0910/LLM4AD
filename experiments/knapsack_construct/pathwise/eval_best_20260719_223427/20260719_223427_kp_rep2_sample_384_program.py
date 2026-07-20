import random
import math
import scipy
try:
    import torch
except Exception:
    torch = None
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
    if not remaining_items:
        return None

    # Fallback: Standard Greedy
    def greedy_select():
        best_item = None
        best_ratio = -1.0
        for weight, value, index in remaining_items:
            if weight <= remaining_capacity:
                ratio = value / weight
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_item = (weight, value, index)
        return best_item

    # Filter fitting items
    fitting_items = [item for item in remaining_items if item[0] <= remaining_capacity]
    if not fitting_items:
        return None

    # Pre-sort items by weight for efficient residual checking
    items_by_weight = sorted(remaining_items, key=lambda x: x[0])
    
    # Calculate statistics for dynamic multiplier and thresholds
    weights = [w for w, _, _ in remaining_items]
    values = [v for _, v, _ in remaining_items]
    
    if not weights or not values:
        return greedy_select()

    avg_weight = sum(weights) / len(weights)
    
    # Calculate max possible density for normalization (approximated by max ratio among fitting items)
    max_density = -1.0
    for w, v, _ in fitting_items:
        if w > 0:
            r = v / w
            if r > max_density:
                max_density = r
    
    if max_density == -1.0 or max_density == 0.0:
        return greedy_select()

    # Calculate median fitting weight for stable late-stage normalization
    fitting_weights_list = [w for w, _, _ in fitting_items]
    if len(fitting_weights_list) > 0:
        sorted_fitting_weights = sorted(fitting_weights_list)
        nf = len(sorted_fitting_weights)
        if nf % 2 == 0:
            median_fitting_weight = (sorted_fitting_weights[nf // 2 - 1] + sorted_fitting_weights[nf // 2]) / 2.0
        else:
            median_fitting_weight = sorted_fitting_weights[nf // 2]
    else:
        median_fitting_weight = avg_weight

    best_item = None
    best_score = -float('inf')
    
    # Refined Core Decision Rule: Median-Normalized Waste Penalty + Simplified Look-Ahead
    for weight, value, index in fitting_items:
        ratio = value / weight
        residual = remaining_capacity - weight
        
        # 1. Waste Penalty using Median Normalization
        # Penalize large residuals relative to the median fitting weight, ensuring stability in late stages.
        waste_penalty = 0.0
        if residual > 0:
            # Estimate how well the residual can be filled
            best_residual_ratio = -1.0
            for w, v, idx in items_by_weight:
                if w > residual:
                    break
                if idx == index:
                    continue
                if w > 0:
                    r = v / w
                    if r > best_residual_ratio:
                        best_residual_ratio = r
            
            # If no item fits in residual, it's pure waste
            if best_residual_ratio <= 0:
                fill_efficiency = 0.0
            else:
                fill_efficiency = best_residual_ratio / max_density
            
            # Log-scaled residual penalty normalized by median fitting weight
            log_factor = math.log(1.0 + residual / (median_fitting_weight + 1e-9))
            
            # Harmonic penalty component
            if fill_efficiency > 0:
                harmonic_term = 1.0 / (1.0 + fill_efficiency)
                waste_penalty = harmonic_term * log_factor * max_density
            else:
                # Pure waste case
                waste_penalty = log_factor * max_density

        # 2. Simplified Look-Ahead Bonus
        # Reward if the residual allows for a high-density follow-up.
        look_ahead_bonus = 0.0
        if residual > 0:
            best_residual_ratio = -1.0
            for w, v, idx in items_by_weight:
                if w > residual:
                    break
                if idx == index:
                    continue
                if w > 0:
                    r = v / w
                    if r > best_residual_ratio:
                        best_residual_ratio = r
            
            if best_residual_ratio > 0:
                # Simplified bonus: scales with residual ratio and follow-up density
                look_ahead_bonus = 0.1 * best_residual_ratio * (residual / (remaining_capacity + 1e-9))

        # 3. Diversity and Exploration (Preserved for stability)
        # Diversity Bonus
        fitting_weights = [w for w, _, _ in fitting_items]
        diversity_bonus = 0.0
        if len(fitting_weights) > 1:
            mean_w_fitting = sum(fitting_weights) / len(fitting_weights)
            variance_w = sum((w - mean_w_fitting)**2 for w in fitting_weights) / len(fitting_weights)
            std_w = variance_w ** 0.5
            if avg_weight > 1e-9:
                diversity_factor = 0.07 * (std_w / avg_weight)
            else:
                diversity_factor = 0.0
            weight_diff = abs(weight - avg_weight) / (avg_weight + 1e-9)
            calculated_bonus = diversity_factor * weight_diff
            cap_bonus = 0.09 * ratio
            diversity_bonus = min(calculated_bonus, cap_bonus)

        # Exploration Bonus (Capacity Variance)
        exploration_bonus = 0.0
        cap_variance_bonus_coeff = 0.02
        cap_var_norm = 0.0
        
        if len(fitting_items) > 1:
            residuals_all = [remaining_capacity - w for w, _, _ in fitting_items]
            mean_residual_all = sum(residuals_all) / len(residuals_all)
            var_residual_all = sum((r - mean_residual_all)**2 for r in residuals_all) / len(residuals_all)
            std_residual_all = var_residual_all ** 0.5
            if std_residual_all > 1e-9:
                cap_var_norm = std_residual_all / (remaining_capacity + 1e-9)
            else:
                cap_var_norm = 0.0
        
        exploration_bonus = cap_variance_bonus_coeff * cap_var_norm * (remaining_capacity / (weight + 1e-9))

        # Final Score Calculation
        score = ratio - waste_penalty + look_ahead_bonus + diversity_bonus + exploration_bonus
        
        if score > best_score:
            best_score = score
            best_item = (weight, value, index)
            
    # Fallback logic
    if best_item is None or best_score < 0:
        return greedy_select()
        
    return best_item

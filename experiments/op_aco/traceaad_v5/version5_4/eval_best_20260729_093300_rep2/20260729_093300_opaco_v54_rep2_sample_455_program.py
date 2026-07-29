import numpy as np
def heuristics(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    """Return edge desirability values for OP ant colony optimization.

    Args:
        prize: Node prizes with shape (n,). Node 0 is the depot.
        distance: Pairwise Euclidean distances with shape (n, n).
            Diagonal entries are large sentinels so self-loops are unused.
        maxlen: Maximum allowed tour length (return-to-depot constrained).

    Returns:
        An (n, n) edge-prior matrix. Larger values make an edge more likely
        to be sampled. Values at or below zero are treated as 1e-9.
    """
    n = prize.shape[0]
    
    # Avoid division by zero
    epsilon = 1e-9
    
    # Compute cost to return to depot from each node j
    # distance[j, 0] is the cost from node j to depot
    return_cost = distance[:, 0]  # shape (n,)
    
    # Compute round-trip cost for each edge (i, j): distance[i, j] + distance[j, 0]
    # return_cost needs to be broadcasted along columns (axis 1)
    # distance is (n, n), return_cost[np.newaxis, :] is (1, n)
    round_trip = distance + return_cost[np.newaxis, :]  # shape (n, n)
    
    # Compute partial cost from depot to current node i
    # distance[0, i] is the cost from depot to node i
    partial_cost = distance[0, :]  # shape (n,)
    
    # Compute remaining budget available for the next edge and return trip
    # remaining = maxlen - partial_cost[i]
    # We need to subtract partial_cost[i] from maxlen for each row i
    # partial_cost[:, np.newaxis] has shape (n, 1)
    remaining_budget = maxlen - partial_cost[:, np.newaxis]  # shape (n, n)
    
    # Ensure remaining budget is not too small or negative to avoid extreme penalties or instability
    # However, if remaining_budget <= round_trip, the edge is infeasible.
    # The term is: exp(-round_trip / (remaining_budget + epsilon))
    # If remaining_budget is small, the exponent is large negative, making the term ~0.
    # If remaining_budget is negative (infeasible start), we clamp it to a small positive value
    # to keep the term finite but very small.
    
    # Calculate denominator for the budget term
    denom = remaining_budget + epsilon
    denom = np.maximum(denom, epsilon) # Prevent division by zero or extremely small numbers
    
    # Local remaining-budget-aware budget term
    budget_term = np.exp(-round_trip / denom)
    
    # Dynamic normalized prize-to-distance ratio: prize[j] / distance[i, j]
    # prize is (n,), distance is (n, n)
    # We need prize[j] for each column j. So prize[np.newaxis, :] is (1, n)
    prize_row = prize[np.newaxis, :]  # (1, n)
    
    # Clip distance to avoid division by zero, though sentinels are large
    dist_clipped = np.maximum(distance, epsilon)
    
    prize_dist_ratio = prize_row / dist_clipped  # shape (n, n)
    
    # Scaled and clamped dynamic exponent: uses prize_dist_ratio * maxlen to scale
    # adaptively based on global budget and prize magnitude, clipped to [1.0, 4.0]
    # Retained from Primary Program Step 7 / Reference Step 5
    dynamic_exponent = np.clip(1.5 * np.log1p(prize_dist_ratio * maxlen), 1.0, 4.0)
    
    # Combine: ratio^dynamic_exponent * budget_term
    # To compute base^exponent efficiently: exp(exponent * log(base))
    # base = prize_dist_ratio
    # exponent = dynamic_exponent
    # We need to handle cases where prize_dist_ratio might be 0 (though clipped)
    # log(base) could be -inf if base is 0, but we clipped base to >= epsilon
    log_ratio = np.log(prize_dist_ratio)
    
    # Compute ratio_power = exp(dynamic_exponent * log_ratio)
    ratio_power = np.exp(dynamic_exponent * log_ratio)
    
    # Combine with budget term
    heuristic_matrix = ratio_power * budget_term
    
    # Ensure no negative or zero values (treated as 1e-9 by ACO)
    # The budget term is >= 0. The ratio term is >= 0.
    # We clamp to epsilon to ensure strictly positive finite values as per contract.
    heuristic_matrix = np.maximum(heuristic_matrix, epsilon)
    
    return heuristic_matrix

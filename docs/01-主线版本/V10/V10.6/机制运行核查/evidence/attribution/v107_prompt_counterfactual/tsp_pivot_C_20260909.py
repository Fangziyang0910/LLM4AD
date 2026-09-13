import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Selects the next node using a 2-step look-ahead heuristic that balances
    immediate distance, future exit options, and return to destination.
    """
    if len(unvisited_nodes) == 0:
        return destination_node

    n_remaining = len(unvisited_nodes)

    if n_remaining == 1:
        return int(unvisited_nodes[0])

    # Distance from current node to all candidates
    d_curr = distance_matrix[current_node, unvisited_nodes]
    
    # Distance from candidates to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Submatrix of distances among unvisited nodes
    sub = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # 2-step lookahead:
    # If we pick node i, what is the minimum cost to leave it?
    # We can either go to another unvisited node, or to the destination.
    # Note: If n_remaining > 1, we can go to another unvisited node.
    # If n_remaining == 1, we must go to destination.
    
    # For each candidate i, the best next move is min(min distance to other unvisited, distance to destination)
    # Since sub includes self-distances (0), we should exclude self when finding min to "other" unvisited.
    # However, distance to self is 0. If we just take min(sub[i]), it will be 0. 
    # We need min(distance to OTHER unvisited nodes).
    
    if n_remaining > 1:
        # Mask diagonal to ignore self
        sub_no_self = sub.copy()
        np.fill_diagonal(sub_no_self, np.inf)
        min_exit_to_unvisited = np.min(sub_no_self, axis=1)
        
        # The best exit is either to another unvisited node OR to the destination
        # But wait, if we go to destination, we are done. So the "cost" of exiting to destination is d_dest.
        # The "cost" of exiting to another unvisited node is min_exit_to_unvisited.
        # So the 2nd step cost is min(min_exit_to_unvisited, d_dest)
        second_step_cost = np.minimum(min_exit_to_unvisited, d_dest)
    else:
        second_step_cost = d_dest.copy()

    # Calculate progress (0 at start, 1 at end)
    n_total = distance_matrix.shape[0]
    if n_total <= 1:
        return destination_node
    progress = (n_total - n_remaining) / (n_total - 1)
    
    # Dynamic weight for return to destination
    # Early on (progress ~ 0), we care mostly about local moves (greedy + lookahead).
    # Late on (progress ~ 1), we strongly prefer nodes close to destination.
    # Use a smooth curve: 0.1 + 0.9 * smoothstep(progress)
    # smoothstep is 0 for p=0, 1 for p=1, with zero derivative at ends.
    smooth_p = progress * progress * (3 - 2 * progress)
    return_weight = 0.1 + 0.8 * smooth_p
    
    # Score calculation
    # Primary: Immediate distance
    # Secondary: Lookahead exit cost (discounted by 0.5 to prioritize immediate step but consider future)
    # Tertiary: Return to destination (weighted by progress)
    
    # Normalize terms to avoid scale issues? 
    # The distance matrix is fixed, so raw distances are comparable within the instance.
    # However, mixing d_curr (large) with d_dest (large) is fine.
    # We want to minimize: d_curr + 0.5 * second_step_cost + return_weight * d_dest
    
    # A more robust approach might be to weight the second step lower if we are far from the end,
    # but the lookahead itself is a heuristic for "connectivity".
    
    scores = d_curr + 0.5 * second_step_cost + return_weight * d_dest
    
    # Handle edge cases (NaN/Inf)
    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
        scores = d_curr + 1e-9 * np.arange(n_remaining)

    idx = int(np.argmin(scores))
    return int(unvisited_nodes[idx])
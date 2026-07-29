import numpy as np
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 0:
        # No unvisited nodes, return to destination/start
        return destination_node
    
    if len(unvisited_nodes) == 1:
        # Only one unvisited node, visit it
        return int(unvisited_nodes[0])
    
    # Calculate distances from current node to all unvisited candidates
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Calculate distances from each unvisited candidate to the destination node
    dist_to_destination = distance_matrix[unvisited_nodes, destination_node]
    
    # Calculate pairwise distances among unvisited nodes
    pairwise_dists = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Replace diagonal with infinity to exclude self-distance when finding min
    np.fill_diagonal(pairwise_dists, np.inf)
    
    # Find nearest and second-nearest neighbor distances for each unvisited node
    # Sort distances for each row to get the two smallest values
    sorted_dists = np.sort(pairwise_dists, axis=1)
    
    min_dist_to_others = sorted_dists[:, 0]
    second_min_dist_to_others = sorted_dists[:, 1]
    
    # Dynamic K based on average distance to destination
    # This scales the sigmoid transition based on geometric spread relative to return path
    n_total = distance_matrix.shape[0]
    remaining = len(unvisited_nodes)
    avg_dist_to_dest = np.mean(dist_to_destination)
    # Scale K by avg distance to dest relative to a baseline (e.g., mean of all dists to dest)
    # Use a small epsilon to avoid division by zero
    baseline_avg_dist = np.mean(distance_matrix[:, destination_node])
    K = (n_total / 4.0) * (avg_dist_to_dest / (baseline_avg_dist + 1e-9))
    
    # Inverse sigmoid weight for connectivity:
    # c starts low when many nodes remain (low cluster pressure) and increases as few remain (high cluster pressure)
    # Formula: c = 0.2 + 0.3 * (K / (remaining + K))
    c = 0.2 + 0.3 * (K / (remaining + K))
    
    # Dynamic hybrid connectivity weights based on cluster pressure c
    # Early tour (low c): w1 high (prioritize min_dist), w2 low
    # Late tour (high c): w1 low, w2 high (prioritize second_min_dist for robustness)
    w1 = 0.5 + 0.25 * (1 - c)
    w2 = 0.5 - 0.25 * (1 - c)
    
    hybrid_conn_metric = w1 * min_dist_to_others + w2 * second_min_dist_to_others

    # Weight for the destination distance term
    w = 0.5
    
    # Score = dist_from_current - w * dist_to_destination - c * hybrid_conn_metric
    # We want to minimize this score.
    # - dist_from_current: prefer close nodes (minimize)
    # - - w * dist_to_destination: prefer nodes far from destination (subtracting larger value lowers score)
    # - - c * hybrid_conn_metric: prefer nodes in dense clusters (subtracting larger value lowers score)
    
    scores = dist_from_current - w * dist_to_destination - c * hybrid_conn_metric
    
    # Select the candidate with the minimum score
    min_idx = np.argmin(scores)
    next_node = int(unvisited_nodes[min_idx])
    
    return next_node

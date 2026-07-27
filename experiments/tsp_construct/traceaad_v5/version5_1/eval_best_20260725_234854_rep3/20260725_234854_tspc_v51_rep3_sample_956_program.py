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
        return destination_node
    
    # If only one unvisited node left (besides destination), pick it
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Extract distances from current node to all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Extract distances between all pairs of unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Calculate destination-aware weighted isolation
    # Formula: mean_dist_weighted = (sum(distances to other unvisited) + 2.0 * dist_to_destination) / (n_unvisited + 2.0)
    # Reverted destination weight coefficient and denominator offset to 2.0 for stability
    n_unvisited = len(unvisited_nodes)
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    mean_dist_weighted = (np.sum(sub_matrix, axis=1) + 2.0 * dist_to_dest) / (n_unvisited + 2.0)
    
    # Dynamic convexity parameter: scales from ~2.0 (early, isolation-focused) to ~1.0 (late, distance-focused)
    # p = 1.0 + (n_unvisited / (n_unvisited + 5.0))
    # When n_unvisited is large: p -> 1.0 + 1.0 = 2.0
    # When n_unvisited is small: p -> 1.0 + 0 = 1.0
    p = 1.0 + (n_unvisited / (n_unvisited + 5.0))
    
    # Raw weighted isolation term with dynamic exponent to preserve absolute magnitudes
    epsilon = 1e-8
    inverse_norm_term = 1.0 / (np.power(mean_dist_weighted, p) + epsilon)
    
    # The score combines immediate distance with an isolation bonus
    score = dist_from_current * inverse_norm_term
    
    # Select node with minimum score
    best_idx = np.argmin(score)
    
    return int(unvisited_nodes[best_idx])

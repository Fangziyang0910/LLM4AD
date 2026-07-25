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
    
    # If only one unvisited node, must go there
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_candidates = len(unvisited_nodes)
    
    # 1. Distance from current node to candidate
    dist_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Efficiently extract the submatrix of distances between all unvisited nodes.
    # This matrix is used for both peripherality calculation.
    indices = unvisited_nodes
    dist_unvisited = distance_matrix[np.ix_(indices, indices)]
    
    # 3. Connectivity/Peripherality Score: Average distance from candidate to other unvisited nodes.
    # High mean distance indicates the node is peripheral/isolated in the current set.
    # Sum of distances / (N-1) to exclude self-distance (diagonal is 0).
    sum_dist_to_others = np.sum(dist_unvisited, axis=1)
    mean_dist_to_others = sum_dist_to_others / (n_candidates - 1)
    
    # 4. Normalize peripherality by global mean distance of the candidate set
    # This scales the geometric penalty relative to the instance's overall density.
    # Total sum of all distances in the submatrix divided by number of pairs (N*(N-1))
    total_dist = np.sum(dist_unvisited)
    global_mean_dist = total_dist / (n_candidates * (n_candidates - 1))
    
    # Avoid division by zero in degenerate cases
    if global_mean_dist == 0:
        global_mean_dist = 1e-9
        
    normalized_periph = mean_dist_to_others / global_mean_dist

    # 5. Scale by max peripherality among candidates
    # This ensures the penalty term is bounded and robust to outliers.
    max_periph = np.max(normalized_periph)
    if max_periph == 0:
        max_periph = 1e-9
        
    scaled_periph = normalized_periph / max_periph
    
    # 6. Distance from candidate to destination node
    dist_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 7. Compute Adaptive Weights
    # Alpha: Adaptive weight for destination proximity. Increases as n_candidates decreases.
    # This encourages steering towards the end of the tour as it progresses.
    alpha = 0.5 / (n_candidates + 1)
    
    # Beta: Dynamic weight for peripherality preference.
    # Use quadratic decay from primary (p277) for sharper reduction in later stages,
    # prioritizing distance minimization for tight closure.
    n_total = distance_matrix.shape[0]
    if n_total <= 0:
        n_total = 1
        
    beta = 0.5 * (n_candidates / n_total) ** 0.5
    
    # 8. Compute Adaptive Power-Law Exponent for Destination Attraction
    # Adopted from reference (p275): Logarithmic decay for smoother transition.
    # p transitions from 0.5 (early tour, sqrt, robust) to 1.0 (late tour, linear inverse, precise)
    if n_total <= 1:
        p = 0.5
    else:
        p = 0.5 + 0.5 * np.log(n_total / n_candidates) / np.log(n_total)
        
    # Clamp p to [0.5, 1.0] for safety
    p = np.clip(p, 0.5, 1.0)
    
    # 9. Composite Score
    # Lower score is better.
    # Proximity (dist_current) + Adaptive Power-Law Destination Attraction - Local Geometry Reward (beta * scaled_periph)
    
    # Adaptive epsilon: Scale with mean distance to destination to maintain stability across different instance scales.
    # A smaller scaling factor (0.001) was chosen to avoid the regression seen with 0.01, preserving the relative 
    # strength of the attraction term while preventing numerical issues.
    eps = np.mean(dist_dest) * 0.001 + 1e-9
    
    # Adaptive power-law attraction: alpha / (dist_dest + eps)^p
    dest_attraction = alpha / np.power(dist_dest + eps, p)
    
    scores = dist_current + dest_attraction - beta * scaled_periph
    
    # Find index of minimum score
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])

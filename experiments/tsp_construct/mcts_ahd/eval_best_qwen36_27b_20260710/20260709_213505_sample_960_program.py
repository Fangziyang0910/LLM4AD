
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
    import numpy as np

    if len(unvisited_nodes) == 0:
        return -1
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    candidates = np.asarray(unvisited_nodes, dtype=int)
    n_candidates = len(candidates)
    
    # Distances from current node to all candidates
    dist_current_to_candidates = distance_matrix[current_node, candidates]
    
    # Distance from destination to all candidates
    dist_dest_to_candidates = distance_matrix[destination_node, candidates]
    
    # Pre-fetch submatrix for efficient computation of local/global metrics
    dists_among_unvisited = distance_matrix[np.ix_(candidates, candidates)]
    
    epsilon = 1e-9
    
    # --- Component 1: Immediate Distance Cost ---
    # Use power 1.2 for smoother gradients (inspired by No.1's improved performance)
    dist_cost = dist_current_to_candidates ** 1.2
    
    # --- Component 2: Dynamic Destination Repulsion ---
    # Scale repulsion by mean distance among unvisited nodes and apply decay
    mean_dist_unvisited = np.mean(dists_among_unvisited)
    scale = mean_dist_unvisited if mean_dist_unvisited > epsilon else 1.0
    
    # Decay factor: decreases as n_candidates decreases (inspired by No.2)
    decay_factor = np.clip(0.3 + 0.7 * (n_candidates / 20.0), 0.1, 1.0)
    
    # Repulsion term: higher when node is close to destination
    repulsion_term = scale * decay_factor / (dist_dest_to_candidates + epsilon)
    
    # --- Component 3: Weighted Linear Tension Factor (inspired by No.1) ---
    
    # 1. Global Isolation: Mean distance to all other unvisited nodes
    sum_dist_to_others = np.sum(dists_among_unvisited, axis=1)
    mean_dist_to_others = sum_dist_to_others / (n_candidates - 1)
    
    # 2. Local Vulnerability: Distance to nearest unvisited neighbor
    dists_among_unvisited_no_self = dists_among_unvisited.copy()
    np.fill_diagonal(dists_among_unvisited_no_self, np.inf)
    min_dist_to_nearest = np.min(dists_among_unvisited_no_self, axis=1)
    min_dist_to_nearest = np.where(min_dist_to_nearest == np.inf, 0.0, min_dist_to_nearest)
    
    # Normalize both metrics to [0, 1] range
    max_mean = np.max(mean_dist_to_others)
    min_mean = np.min(mean_dist_to_others)
    range_mean = max_mean - min_mean
    
    if range_mean < epsilon:
        norm_mean_dist = np.zeros_like(mean_dist_to_others)
    else:
        norm_mean_dist = (mean_dist_to_others - min_mean) / range_mean
        
    max_min_dist = np.max(min_dist_to_nearest)
    min_min_dist = np.min(min_dist_to_nearest)
    range_min_dist = max_min_dist - min_min_dist
    
    if range_min_dist < epsilon:
        norm_min_dist = np.zeros_like(min_dist_to_nearest)
    else:
        norm_min_dist = (min_dist_to_nearest - min_min_dist) / range_min_dist
        
    # Combine using weighted linear combination (0.8/0.2) inspired by No.1
    w1 = 0.8
    w2 = 0.2
    tension_linear = w1 * norm_mean_dist + w2 * norm_min_dist
    
    # Apply non-linearity to sharpen preference for truly isolated nodes
    tension_score = np.power(tension_linear, 1.5)
    
    # --- Horizon-Aware Topology Emphasis (inspired by No.1) ---
    
    # Calculate horizon ratio: distance from current to destination vs mean spread
    dist_current_to_dest = distance_matrix[current_node, destination_node]
    
    if mean_dist_unvisited < epsilon:
        horizon_ratio = 1.0
    else:
        horizon_ratio = dist_current_to_dest / mean_dist_unvisited
        
    # Clip horizon ratio to prevent extreme values
    horizon_ratio = np.clip(horizon_ratio, 0.0, 2.0)
    
    # Topology Emphasis Factor: when far from destination, emphasize isolation more
    topology_emphasis = 1.0 + 1.5 * tension_score * np.clip(horizon_ratio, 0.0, 1.0)
    
    # --- Final Score Calculation ---
    # Score = (DistCost * Repulsion) / TopologyEmphasis
    # We want to MINIMIZE this score.
    
    total_scores = (dist_cost * repulsion_term) / topology_emphasis
    
    # Select the candidate with the minimum score
    best_idx = np.argmin(total_scores)
    next_node = int(candidates[best_idx])
    
    return next_node

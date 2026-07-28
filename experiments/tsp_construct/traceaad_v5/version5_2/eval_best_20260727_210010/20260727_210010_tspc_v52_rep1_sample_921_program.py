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
    
    # If only one unvisited node, pick it
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_unvisited = len(unvisited_nodes)
    
    # Precompute distances from current node to all unvisited nodes
    current_to_candidates = distance_matrix[current_node][unvisited_nodes]
    
    # Extract the sub-matrix of distances among unvisited nodes for efficiency
    # This allows vectorized calculation of avg distances
    unvisited_dist_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Compute average distance from each candidate to all other unvisited nodes (excluding self)
    # This serves as a proxy for "Strandedness Priority":
    # Nodes with high average distance to others are geometrically peripheral/stranded.
    # Sum of distances to all others
    sum_dists = np.sum(unvisited_dist_matrix, axis=1)
    
    # Average distance (excluding self, which is 0 in distance matrix usually, but safe to subtract if not)
    # Since diagonal is 0, sum_dists[i] is sum to all others.
    avg_dists = sum_dists / (n_unvisited - 1)
    
    # Dynamic power scaling based on phase
    # Early stage (>10 nodes): higher exponent (1.5) to aggressively penalize peripheral nodes
    # Late stage (<=10 nodes): lower exponent (1.0) to rely on raw distance discrimination
    power_exponent = 1.5 if n_unvisited > 10 else 1.0
    avg_dists_scaled = avg_dists ** power_exponent
            
    # Local Min-Max normalization for stability and relative sensitivity
    # Normalize immediate distance
    min_immediate = np.min(current_to_candidates)
    max_immediate = np.max(current_to_candidates)
    if max_immediate > min_immediate:
        normalized_dist = (current_to_candidates - min_immediate) / (max_immediate - min_immediate)
    else:
        normalized_dist = np.zeros_like(current_to_candidates)
        
    # Normalize strandedness (avg distance) using scaled values
    min_avg = np.min(avg_dists_scaled)
    max_avg = np.max(avg_dists_scaled)
    if max_avg > min_avg:
        normalized_avg = (avg_dists_scaled - min_avg) / (max_avg - min_avg)
    else:
        normalized_avg = np.zeros_like(avg_dists_scaled)
            
    # Global normalization for stability as per Experience 3 was replaced by local min-max
    
    # Compute distances from candidates to destination (raw)
    dists_to_dest = distance_matrix[destination_node][unvisited_nodes]
    
    # Phase-dependent static heuristic for alpha
    # alpha = 0.6 when n_unvisited <= 5 (late stage, aggressive strandedness), else 0.3
    alpha = 0.6 if n_unvisited <= 5 else 0.3
    
    # Core score: immediate distance - strandedness benefit
    # We want to minimize immediate distance, maximize strandedness (subtract)
    core_scores = normalized_dist - alpha * normalized_avg
    
    # Adaptive destination penalty based on geometric spread
    # Scale by average distance among unvisited nodes to adjust for density
    # Prevent division by zero by adding a small epsilon if avg_dists is all 0 (should not happen if n > 1)
    mean_avg_dist = np.mean(avg_dists)
    if mean_avg_dist <= 0:
        mean_avg_dist = 1e-9
        
    adaptive_factor = np.exp(dists_to_dest / (n_unvisited * mean_avg_dist))
    
    scores = core_scores * adaptive_factor
    
    # Select candidate with minimum score
    best_idx = np.argmin(scores)
    best_node = unvisited_nodes[best_idx]
    
    return int(best_node)

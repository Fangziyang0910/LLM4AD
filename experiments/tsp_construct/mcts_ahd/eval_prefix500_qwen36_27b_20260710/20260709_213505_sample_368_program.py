
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

    candidates = unvisited_nodes
    
    # Calculate distances from current node to each candidate
    dist_from_current = np.array([
        distance_matrix[current_node, node]
        for node in candidates
    ])
    
    # Calculate distances from each candidate to destination
    dist_to_destination = np.array([
        distance_matrix[node, destination_node]
        for node in candidates
    ])
    
    # Direct distance from current to destination
    d_current_dest = distance_matrix[current_node, destination_node]
    
    # 1. Base Proximity Term: Squared distance to emphasize short hops
    proximity_term = dist_from_current ** 2
    
    # 2. Alignment Term inspired by No.1: Ratio-based score
    # We use the ratio: (dist_from_current^2) / (dist_to_destination + epsilon)
    # This rewards nodes that are close to current AND close to destination.
    epsilon = 1e-6
    ratio_score = proximity_term / (dist_to_destination + epsilon)
    
    # 3. Local Density Term
    # High density means nodes are close together. 
    # We want to penalize nodes in dense clusters early to avoid getting stuck?
    # Use K=3 for local density calculation
    K = min(3, len(candidates) - 1)
    density_penalty = np.ones(len(candidates))
    
    if K > 0:
        for i, node in enumerate(candidates):
            dists_to_others = []
            for j, other_node in enumerate(candidates):
                if node != other_node:
                    dists_to_others.append(distance_matrix[node, other_node])
            
            dists_sorted = sorted(dists_to_others)
            closest_dists = dists_sorted[:K]
            mean_closest_dist = np.mean(closest_dists)
            
            # Penalty is inverse of mean distance. 
            # High density -> low mean distance -> high penalty.
            # Add epsilon for stability.
            density_penalty[i] = 1.0 / (mean_closest_dist + 1e-6)

    # 4. Dynamic Weighting based on Progress
    try:
        max_n = distance_matrix.shape[0]
        n_unvisited = len(unvisited_nodes)
        progress = 1.0 - (n_unvisited / max_n)
    except:
        progress = 0.0
    
    # Weight for structural penalties decays as progress increases.
    # Using (1 - progress) ensures structural importance at start.
    # Using power 1.5 for steeper decay than linear.
    structural_weight = (1.0 - progress) ** 1.5
    
    # 5. Composite Score
    # Base cost is the ratio score (No.1 inspiration).
    # Add penalties scaled by structural weight.
    
    # Enhance alignment weight with a small constant to ensure some directional bias always exists
    w_ratio = 1.0
    w_d = structural_weight * 0.8  # Slightly less weight for density to avoid over-penalizing
    
    # Multiply the density penalty by the ratio score to keep units consistent 
    # and ensure that for very close nodes, the absolute penalty contribution is small 
    # unless density is extreme.
    
    scores = (w_ratio * ratio_score + 
              w_d * density_penalty * ratio_score)
    
    best_index = np.argmin(scores)
    next_node = int(candidates[best_index])
    
    return next_node

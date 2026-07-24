import numpy as np

def update_edge_distance(edge_distance: np.ndarray, local_opt_tour: np.ndarray, edge_n_used: np.ndarray) -> np.ndarray:
    """
    Design a novel algorithm to update the distance matrix.

    Args:
    edge_distance: A matrix of the distance.
    local_opt_tour: An array of the local optimal tour of IDs.
    edge_n_used: A matrix of the number of each edge used during permutation.

    Return:
    updated_edge_distance: A matrix of the updated distance.
    """
    n = edge_distance.shape[0]
    updated_distance = edge_distance.copy()
    
    if len(local_opt_tour) < 2:
        return updated_distance
    
    # Extract edges from the local optimal tour
    tour_edges = []
    tour_edge_distances = []
    for i in range(len(local_opt_tour)):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % len(local_opt_tour)]
        tour_edges.append((u, v))
        tour_edge_distances.append(edge_distance[u, v])
    
    tour_edge_distances = np.array(tour_edge_distances)
    
    # Calculate the total distance of the current tour
    tour_length = np.sum(tour_edge_distances)
    
    # Calculate the average distance of all edges in the matrix
    non_diag_mask = ~np.eye(n, dtype=bool)
    non_diag_distances = edge_distance[non_diag_mask]
    
    if len(non_diag_distances) > 0:
        avg_distance = np.mean(np.abs(non_diag_distances))
    else:
        avg_distance = 1.0
        
    if avg_distance == 0:
        avg_distance = 1.0
        
    # Calculate standard deviation of tour edge distances
    if len(tour_edge_distances) > 1:
        std_tour = np.std(tour_edge_distances)
    else:
        std_tour = 0.0
        
    # Dynamic scaling factor based on problem geometry
    dynamic_scale_factor = tour_length / avg_distance
    
    # Step 1: Apply adaptive global decay to all edges towards average
    # Edges with high penalties (distance >> avg_distance) decay faster
    base_decay = 0.95
    min_decay = 0.80 # Prevent decay from becoming too aggressive
    
    # Calculate deviation from average
    deviation = updated_distance - avg_distance
    
    # Calculate adaptive decay factor for each edge
    # penalty_ratio is roughly (distance - avg) / avg
    penalty_ratio = deviation / avg_distance
    
    # Decay factor decreases as penalty_ratio increases
    # decay = base_decay - adjustment * penalty_ratio
    # Clamp penalty_ratio to reasonable values to prevent extreme decay
    # Only apply acceleration to positive deviations (penalties)
    effective_ratio = np.maximum(0, penalty_ratio)
    adaptive_decay = base_decay - 0.05 * effective_ratio
    adaptive_decay = np.clip(adaptive_decay, min_decay, 1.0)
    
    # Apply adaptive decay: move distance towards avg_distance
    # updated_distance = avg_distance + (distance - avg_distance) * decay
    updated_distance = avg_distance + deviation * adaptive_decay
    
    # Step 2: Penalize edges in the current tour
    # Penalty is inversely proportional to usage count and scaled by variability
    for u, v in tour_edges:
        count = edge_n_used[u, v]
        
        # Base penalty inversely proportional to usage count
        base_penalty = dynamic_scale_factor * (1.0 / (1.0 + count))
        
        # Variability factor: Scale penalty by how much the tour edges vary
        # Adding 1.0 ensures a baseline penalty even if std is 0
        # Normalizing std_tour by avg_distance makes it dimensionless
        variability_factor = 1.0 + (std_tour / avg_distance)
        
        penalty_magnitude = base_penalty * variability_factor
        
        updated_distance[u, v] += penalty_magnitude
        updated_distance[v, u] += penalty_magnitude
        
    # Ensure distances remain positive
    updated_distance = np.maximum(updated_distance, 0)
    
    return updated_distance

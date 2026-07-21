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
    updated_distance = edge_distance.copy()
    
    n_cities = len(local_opt_tour)
    if n_cities == 0:
        return updated_distance

    # Calculate the current best tour's total distance (fitness)
    current_tour_distance = 0.0
    for i in range(n_cities):
        current_city = local_opt_tour[i]
        next_city = local_opt_tour[(i + 1) % n_cities]
        current_tour_distance += edge_distance[current_city, next_city]
    
    # Avoid division by zero
    if current_tour_distance == 0:
        current_tour_distance = 1.0
    
    # Calculate the average possible edge distance in the matrix for global scaling
    avg_matrix_distance = np.mean(edge_distance)
    if avg_matrix_distance == 0:
        avg_matrix_distance = 1.0
    
    # Global scaling factor: ratio of current tour distance to average matrix distance
    global_scale_base = np.sqrt(current_tour_distance / avg_matrix_distance)
    
    # --- Dynamic Stagnation Estimation using Shannon Entropy ---
    
    # Flatten the usage matrix to calculate entropy
    usage_counts = edge_n_used.flatten()
    total_usage = np.sum(usage_counts)
    
    if total_usage == 0:
        # If no edges have been used, entropy is 0 (or undefined, we treat as 0 stagnation/uncertainty)
        normalized_entropy = 0.0
    else:
        # Calculate probabilities
        p = usage_counts / total_usage
        
        # Calculate Shannon entropy: H = -sum(p * log(p))
        # Handle 0*log(0) = 0
        entropy_term = np.where(p > 0, p * np.log(p), 0.0)
        shannon_entropy = -np.sum(entropy_term)
        
        # Determine the support size (number of non-zero edges) for normalization
        # Using the number of non-zero entries ensures normalized entropy is in [0, 1]
        num_non_zero = np.sum(usage_counts > 0)
        if num_non_zero <= 1:
            max_entropy = 1.0 # Avoid log(1)=0 division error, though entropy is 0 if 1 or 0 items
        else:
            max_entropy = np.log(num_non_zero)
            
        # Avoid division by zero if max_entropy is 0
        if max_entropy == 0:
            normalized_entropy = 0.0
        else:
            normalized_entropy = shannon_entropy / max_entropy
            
    # Decay Rate hyperparameter
    decay_rate = 0.5
    
    # Calculate decay factor based on normalized entropy
    # High entropy -> smaller decay factor -> reduced penalty
    decay_factor = np.exp(-normalized_entropy * decay_rate)
    
    # Adjusted global scale factor
    global_scale_factor = global_scale_base * decay_factor
    
    # Define a small constant to avoid division by zero in usage count
    eps = 1e-6
    
    # Calculate edge-specific repulsion potential for edges in the tour
    for i in range(n_cities):
        current_city = local_opt_tour[i]
        next_city = local_opt_tour[(i + 1) % n_cities]
        
        # Get usage count for this edge
        usage_count = edge_n_used[current_city, next_city]
        
        # Calculate inverse usage potential
        inverse_usage = 1.0 / (usage_count + eps)
        
        # Calculate the repulsion potential for this edge
        repulsion_potential = global_scale_factor * inverse_usage
        
        # Update the distance by adding the repulsion potential
        updated_distance[current_city, next_city] += repulsion_potential
        # Since it's likely an undirected graph, update the reverse edge too
        updated_distance[next_city, current_city] += repulsion_potential
    
    # Ensure no negative distances
    updated_distance = np.maximum(updated_distance, 0)
    
    return updated_distance

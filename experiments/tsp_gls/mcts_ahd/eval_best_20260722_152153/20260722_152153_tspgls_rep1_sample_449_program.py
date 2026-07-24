
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
    
    n = len(local_opt_tour)
    
    if n <= 1:
        return updated_distance

    # Extract edges in the current local optimal tour
    # u: tour[i], v: tour[(i+1)%n]
    u = local_opt_tour
    v = np.roll(local_opt_tour, -1)
    
    # Get usage counts for these edges
    usage_counts = edge_n_used[u, v]
    
    # Calculate robust local statistics
    median_usage = np.median(usage_counts)
    mad = np.mean(np.abs(usage_counts - median_usage))
    
    epsilon = 1e-9
    
    # Avoid division by zero in MAD
    if mad < epsilon:
        mad = 1.0
        
    # Robust Z-score
    z_local = (usage_counts - median_usage) / mad
    
    # Logistic transformation parameters
    # Steeper sigmoid to more sharply distinguish overused edges
    logistic_steepness = 3.0
    # Map z to [0, 1]
    logistic_factor = 1.0 / (1.0 + np.exp(-logistic_steepness * z_local))
    # Map to penalty multiplier range. 
    # We allow a wider range for more aggressive exploration when needed.
    # Range [0.2, 2.8] means min penalty is 20% of base, max is 280%.
    penalty_logistic = 0.2 + 2.6 * logistic_factor 
    
    # Global Context
    total_global_usage = np.sum(edge_n_used)
    if total_global_usage < epsilon:
        total_global_usage = 1.0

    # Global Frequency Damping
    # Inverse square root of global frequency
    global_freq = usage_counts / (total_global_usage + epsilon)
    damping_global = 1.0 / (np.sqrt(global_freq + epsilon))
    
    # Node Centrality Damping
    # Sum of usage in and out for each node
    node_importance = np.sum(edge_n_used, axis=1) + np.sum(edge_n_used, axis=0)
    max_node_imp = np.max(node_importance)
    if max_node_imp < epsilon:
        max_node_imp = 1.0
    normalized_node_imp = node_importance / max_node_imp
    
    # Edge centrality is the average of the two nodes' importance
    edge_centrality = (normalized_node_imp[u] + normalized_node_imp[v]) / 2.0
    # Inverse linear damping: lower centrality -> higher penalty factor
    # Add epsilon to avoid division by zero if centrality is 0
    damping_centrality = 1.0 / (edge_centrality + epsilon)
    
    # Combine terms
    lambda_base = 0.12
    
    # Penalty Magnitude
    penalty_magnitude = lambda_base * penalty_logistic * damping_global * damping_centrality
    
    # Original distances for scaling
    original_dist = edge_distance[u, v]
    
    # Calculate added penalties
    added_penalty = original_dist * penalty_magnitude
    
    # Update distances (symmetric matrix)
    np.add.at(updated_distance, (u, v), added_penalty)
    np.add.at(updated_distance, (v, u), added_penalty)
    
    return updated_distance

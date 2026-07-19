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
    if unvisited_nodes.size == 0:
        return destination_node

    num_unvisited = unvisited_nodes.size
    
    # If only one unvisited node, pick it
    if num_unvisited == 1:
        return unvisited_nodes[0]

    # Infer total nodes from the distance matrix shape
    total_nodes = distance_matrix.shape[0]
    
    # Determine k for k-nearest neighbors
    # We use min(10, num_unvisited - 1) to ensure we don't index out of bounds
    # and to focus on local density.
    k = min(10, num_unvisited - 1)

    # Extract the submatrix of distances between unvisited nodes
    # dists[i, j] corresponds to distance between unvisited_nodes[i] and unvisited_nodes[j]
    dists_submatrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Calculate distance from current node to each unvisited node
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Calculate distance from each unvisited node to the destination node
    dists_to_destination = distance_matrix[unvisited_nodes, destination_node]
    
    # Calculate distance from current node to destination node
    dist_current_to_dest = distance_matrix[current_node, destination_node]

    # Calculate local density bonus for each candidate
    # The "density" here is inverse to the average distance to k-nearest neighbors.
    # A node with large average distance to neighbors is "isolated" (sparse region).
    # We want to favor isolated nodes.
    density_scores = np.zeros(num_unvisited)
    
    # Calculate forward-looking connectivity for each candidate
    # This is the minimum distance from the candidate to any other unvisited node.
    # Favoring nodes with low min-distance helps maintain connectivity to the rest of the tour.
    connectivity_scores = np.zeros(num_unvisited)

    # Calculate alignment deviation for each candidate
    # Alignment Deviation = dist(current, node) + dist(node, dest) - dist(current, dest)
    # This represents the detour cost. Smaller detour means better alignment.
    # We subtract this deviation * weight from the score, so smaller detour -> lower score (better)
    alignment_weight = 0.1
    alignment_deviations = dists_from_current + dists_to_destination - dist_current_to_dest

    for i in range(num_unvisited):
        # Get distances from unvisited_nodes[i] to all unvisited_nodes
        row = dists_submatrix[i, :]
        
        # Exclude self-distance (which is 0) by using argsort and skipping the first element
        sorted_indices = np.argsort(row)
        
        # The first element is index i (self), so we skip it.
        # Take the next k elements for density calculation
        if k >= num_unvisited:
            # If k is larger than available neighbors, just take all others
            neighbors = row[sorted_indices[1:]]
        else:
            neighbors = row[sorted_indices[1:k+1]]
            
        density_scores[i] = np.mean(neighbors)

        # For connectivity, the minimum distance to any other node is the second smallest distance in the row
        # since the smallest is self (0).
        connectivity_scores[i] = row[sorted_indices[1]]

    # Define the dynamic weight for the density bonus
    # beta_density scales linearly with the ratio of remaining unvisited nodes to total nodes.
    # Early in the tour (high num_unvisited), beta_density is higher, prioritizing isolated nodes.
    # Late in the tour (low num_unvisited), beta_density is lower, prioritizing direct distance.
    beta_density = 1.0 * (num_unvisited / total_nodes)
    
    # Define the dynamic weight for the connectivity bonus
    # beta_connectivity scales inversely with the number of remaining unvisited nodes.
    # This ensures that as the tour nears completion, connectivity becomes more important
    # to avoid leaving nodes that are far from the current path.
    beta_connectivity = 1.0 / num_unvisited
    
    # Calculate final scores
    # We subtract the bonuses because higher scores (more isolated / better connectivity / better alignment) should lower the total score (make it more attractive)
    # Score = Dist(current, candidate) - beta_density * AvgDist_kNN - beta_connectivity * MinDist_to_Others - beta_alignment * Alignment_Deviation
    scores = dists_from_current - beta_density * density_scores - beta_connectivity * connectivity_scores - alignment_weight * alignment_deviations
    
    # Find the index of the minimum score
    min_idx = np.argmin(scores)
    
    # Get the candidate node ID
    best_candidate = unvisited_nodes[min_idx]
    
    # Handle ties by selecting the node with the smallest ID
    min_score = scores[min_idx]
    epsilon = 1e-9
    
    # Find all candidates with score within epsilon of min_score
    ties_mask = scores <= min_score + epsilon
    ties = unvisited_nodes[ties_mask]
    
    if len(ties) > 1:
        # Select the one with the smallest ID
        best_candidate = ties[np.argmin(ties)]
        
    return best_candidate

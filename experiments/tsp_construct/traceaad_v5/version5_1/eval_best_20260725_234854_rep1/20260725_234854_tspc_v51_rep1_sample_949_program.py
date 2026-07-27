import numpy as np
def _two_opt_convergent(path: list, distance_matrix: np.ndarray) -> list:
    """
    Perform convergent 2-opt local search on a given path until no improving move is found.
    Uses strict Best-Improvement logic: continues until no positive delta (delta > 0) is found in a full pass.
    
    Args:
    path: List of node IDs representing the current tour segment.
    distance_matrix: Distance matrix.

    Returns:
    Improved path list.
    """
    current_path = path[:]
    n = len(current_path)
    
    if n < 3:
        return current_path
        
    while True:
        best_delta = 0
        best_i = -1
        best_j = -1
        
        # Scan all possible 2-opt swaps to find the best improvement
        for i in range(1, n - 1):
            for j in range(i + 1, n - 1):
                # Nodes involved in the swap
                # Segment 1: ... -> path[i-1] -> path[i] ... path[j] -> path[j+1] -> ...
                # We want to reverse segment from i to j
                # Edges to remove: (path[i-1], path[i]) and (path[j], path[j+1])
                # Edges to add: (path[i-1], path[j]) and (path[i], path[j+1])
                
                a, b = current_path[i-1], current_path[i]
                c, d = current_path[j], current_path[j+1]
                
                cost_current = distance_matrix[a, b] + distance_matrix[c, d]
                cost_new = distance_matrix[a, c] + distance_matrix[b, d]
                delta = cost_current - cost_new
                
                if delta > best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        # Termination: if no positive delta found, we are at a local optimum
        if best_delta <= 0:
            break
            
        # Perform the best reversal found
        current_path[best_i:best_j+1] = current_path[best_i:best_j+1][::-1]
            
    return current_path


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
    
    # If two unvisited nodes, evaluate both explicitly using simple logic
    if len(unvisited_nodes) == 2:
        u, v = unvisited_nodes[0], unvisited_nodes[1]
        cost1 = distance_matrix[current_node, u] + distance_matrix[u, v] + distance_matrix[v, destination_node]
        cost2 = distance_matrix[current_node, v] + distance_matrix[v, u] + distance_matrix[u, destination_node]
        return int(u if cost1 < cost2 else v)
    
    n_unvisited = len(unvisited_nodes)
    
    # Fallback for small unvisited sets where regret is less meaningful or computationally expensive relative to benefit
    if n_unvisited < 3:
        dists_from_current = distance_matrix[current_node, unvisited_nodes]
        sorted_by_dist = np.argsort(dists_from_current)
        K = min(20, n_unvisited)
        nearest_indices = sorted_by_dist[:K]
        top_candidates = unvisited_nodes[nearest_indices]
        
        best_candidate = None
        min_lookahead_cost = np.inf
        
        for candidate in top_candidates:
            cost_to_candidate = distance_matrix[current_node, candidate]
            remaining_nodes = unvisited_nodes[unvisited_nodes != candidate]
            
            if len(remaining_nodes) > 0:
                current_sim = candidate
                remaining_sim = remaining_nodes.copy()
                path_order = [candidate]
                
                while len(remaining_sim) > 0:
                    dists_to_remaining = distance_matrix[current_sim, remaining_sim]
                    next_node_idx = np.argmin(dists_to_remaining)
                    next_node = remaining_sim[next_node_idx]
                    current_sim = next_node
                    path_order.append(next_node)
                    remaining_sim = np.delete(remaining_sim, next_node_idx)
                
                full_path = path_order + [destination_node]
                improved_path = _two_opt_convergent(full_path, distance_matrix)
                
                tour_cost = 0.0
                for i in range(len(improved_path) - 1):
                    u_idx = improved_path[i]
                    v_idx = improved_path[i+1]
                    tour_cost += distance_matrix[u_idx, v_idx]
                
                total_estimated_cost = cost_to_candidate + tour_cost
            else:
                total_estimated_cost = cost_to_candidate + distance_matrix[candidate, destination_node]
            
            if total_estimated_cost < min_lookahead_cost:
                min_lookahead_cost = total_estimated_cost
                best_candidate = candidate
                
        return int(best_candidate)

    # 1. Enhanced Regret Calculation with Weighted Destination Penalty
    
    u_nodes = unvisited_nodes
    
    # Create a list of target nodes for neighbor calculation: unvisited + destination
    target_nodes = np.append(u_nodes, destination_node)
    
    # Extract the submatrix of distances from unvisited nodes to target nodes (unvisited + dest)
    # Rows: unvisited nodes, Cols: unvisited nodes + destination
    dists_from_unvisited_to_targets = distance_matrix[np.ix_(u_nodes, target_nodes)]
    
    # Apply weighted destination penalty (beta)
    # The last column corresponds to the destination_node
    beta = 0.5
    dists_masked = dists_from_unvisited_to_targets.copy()
    
    # Multiply the distance to the destination node by beta in the neighbor calculation
    # This makes the destination appear "closer" in the neighbor logic, reducing its priority in regret
    dists_masked[:, -1] *= beta
    
    # Set diagonal to infinity to ignore self-distance for argpartition (only for the unvisited part)
    # The matrix shape is (n_unvisited, n_unvisited + 1)
    # We need to mask the self-distances for the unvisited nodes columns.
    # The destination column (last col) does not need masking against itself since it's not in the row set 'u_nodes'
    
    # Mask the diagonal part corresponding to unvisited nodes visiting themselves
    # Indices 0 to n_unvisited-1 in columns correspond to u_nodes
    np.fill_diagonal(dists_masked[:, :n_unvisited], np.inf)
    
    # Find indices of the 2 smallest values in each row
    # Since we have n_unvisited + 1 columns, we look for 2 nearest neighbors in the augmented set
    partition_indices = np.argpartition(dists_masked, 2, axis=1)[:, :2]
    
    # Extract the actual distances for these indices
    dists_1st = np.take_along_axis(dists_masked, partition_indices[:, [0]], axis=1).flatten()
    dists_2nd = np.take_along_axis(dists_masked, partition_indices[:, [1]], axis=1).flatten()
    
    # Regret is the difference between 2nd nearest and 1st nearest
    regrets = dists_2nd - dists_1st
    
    # 2. Unified Scoring Metric
    # Combine distance from current node and regret into a single score.
    # Score = distance - alpha * regret
    # Lower score is better. High regret (critical nodes) reduces the score, 
    # while short distance also reduces the score.
    
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Normalize regret to be on a similar scale to distance for the alpha factor
    # A simple approach is to use the mean distance among unvisited-to-targets (original, not masked) as a scaling factor
    mean_dist = np.mean(dists_from_unvisited_to_targets[dists_from_unvisited_to_targets < np.inf])
    
    if mean_dist > 0:
        normalized_regrets = regrets / mean_dist
    else:
        normalized_regrets = regrets
        
    # Alpha controls the weight of regret vs distance. 
    # Alpha=1 implies regret and distance are equally important (scaled).
    alpha = 1.0
    
    scores = dists_from_current - alpha * normalized_regrets
    
    # Select top K nodes by lowest score (best combined metric)
    K = min(10, n_unvisited)
    sorted_indices = np.argsort(scores)
    top_candidates_local = sorted_indices[:K]
    
    candidates = u_nodes[top_candidates_local]
    
    # 3. Evaluate candidates with lookahead (NN + 2-opt)
    best_candidate = None
    min_lookahead_cost = np.inf
    
    for candidate in candidates:
        # Cost to move from current node to candidate
        cost_to_candidate = distance_matrix[current_node, candidate]
        
        # Remaining nodes after visiting candidate
        # Create a mask for remaining nodes
        remaining_nodes = unvisited_nodes[unvisited_nodes != candidate]
        
        # Simulate a nearest-neighbor tour for the remaining nodes, then refine with 2-opt
        if len(remaining_nodes) > 0:
            # Build initial sequence: [candidate] + [greedy order of remaining] + [destination]
            # 1. Generate initial path for remaining nodes using Nearest Neighbor starting from candidate
            current_sim = candidate
            remaining_sim = remaining_nodes.copy()
            path_order = [candidate]
            
            while len(remaining_sim) > 0:
                dists_to_remaining = distance_matrix[current_sim, remaining_sim]
                next_node_idx = np.argmin(dists_to_remaining)
                next_node = remaining_sim[next_node_idx]
                
                current_sim = next_node
                path_order.append(next_node)
                
                # Remove visited node from remaining_sim
                remaining_sim = np.delete(remaining_sim, next_node_idx)
            
            # Append destination to complete the segment for 2-opt
            # The path covers: candidate -> ... -> last_remaining -> destination
            full_path = path_order + [destination_node]
            
            # 2. Apply convergent 2-opt refinement on full_path
            improved_path = _two_opt_convergent(full_path, distance_matrix)
            
            # Calculate total cost of the refined path
            tour_cost = 0.0
            for i in range(len(improved_path) - 1):
                u_idx = improved_path[i]
                v_idx = improved_path[i+1]
                tour_cost += distance_matrix[u_idx, v_idx]
            
            total_estimated_cost = cost_to_candidate + tour_cost
        else:
            # No remaining nodes, just go to destination
            total_estimated_cost = cost_to_candidate + distance_matrix[candidate, destination_node]
        
        if total_estimated_cost < min_lookahead_cost:
            min_lookahead_cost = total_estimated_cost
            best_candidate = candidate
            
    return int(best_candidate)

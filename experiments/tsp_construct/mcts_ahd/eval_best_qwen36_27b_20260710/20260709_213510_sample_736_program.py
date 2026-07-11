
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
    
    # Edge case: Only one unvisited node, must visit it
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # Edge case: Only 2 unvisited nodes, use exact evaluation (2 permutations)
    if len(unvisited_nodes) == 2:
        node_a = unvisited_nodes[0]
        node_b = unvisited_nodes[1]
        
        # Path: current -> a -> b -> dest
        cost_a = (distance_matrix[current_node, node_a] + 
                  distance_matrix[node_a, node_b] + 
                  distance_matrix[node_b, destination_node])
        
        # Path: current -> b -> a -> dest
        cost_b = (distance_matrix[current_node, node_b] + 
                  distance_matrix[node_b, node_a] + 
                  distance_matrix[node_a, destination_node])
        
        if cost_a <= cost_b:
            return int(node_a)
        else:
            return int(node_b)

    # Edge case: Only 3 unvisited nodes, use exact evaluation (6 permutations)
    if len(unvisited_nodes) == 3:
        nodes = unvisited_nodes
        best_cost = float('inf')
        best_node = None
        
        # Permutations of the 3 nodes
        import itertools
        for p in itertools.permutations(nodes):
            n1, n2, n3 = p
            # Path: current -> n1 -> n2 -> n3 -> dest
            cost = (distance_matrix[current_node, n1] + 
                    distance_matrix[n1, n2] + 
                    distance_matrix[n2, n3] + 
                    distance_matrix[n3, destination_node])
            if cost < best_cost:
                best_cost = cost
                best_node = n1
                
        return int(best_node)

    # General case: Use composite scoring with look-ahead and stability
    best_node = None
    best_score = float('inf')

    # Pre-fetch distances from current node
    dists_from_current = distance_matrix[current_node]
    
    # Precompute submatrix for unvisited nodes for efficient look-ahead
    unvisited_ids = unvisited_nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_ids, unvisited_ids)]
    n_unvisited = len(unvisited_ids)
    
    # Weights tuning based on analysis of previous algorithms
    # Immediate: Primary driver
    # Lookahead: Strong signal for future efficiency
    # Stability: Prevents structural traps
    # Dest Bias: Ensures connectivity to end
    weight_immediate = 1.0
    weight_lookahead = 0.4      
    weight_stability = 0.15     
    weight_dest_bias = 0.15     

    for i, node in enumerate(unvisited_ids):
        # 1. Immediate Cost
        immediate_dist = dists_from_current[node]
        
        # 2. Structural Stability Penalty
        # Variance of distances from candidate to other unvisited nodes
        # High variance implies the node is either very close to some and far from others,
        # which is risky for TSP tours.
        if n_unvisited > 2:
            # Get distances from this node to all other unvisited nodes
            # sub_matrix[i] contains distances from node i to all nodes in unvisited_ids
            # We exclude the distance to itself (index i in the submatrix row)
            other_dists = np.delete(sub_matrix[i], i)
            
            if len(other_dists) > 1:
                # Use Variance as the primary stability metric
                variance_dist = np.var(other_dists)
                # Normalize variance by the square of the mean to keep it scale-invariant
                mean_dist = np.mean(other_dists)
                if mean_dist > 1e-9:
                    normalized_variance = variance_dist / (mean_dist ** 2)
                else:
                    normalized_variance = 0.0
                
                stability_penalty = weight_stability * normalized_variance
            else:
                stability_penalty = 0.0
        else:
            stability_penalty = 0.0

        # 3. 2-Step Look-Ahead Greedy Cost
        # Simulate a greedy tour for the remaining nodes after picking this one
        # This is more robust than just immediate distance
        lookahead_cost = 0.0
        if n_unvisited > 2:
            # Identify remaining indices
            remaining_indices = [j for j in range(n_unvisited) if j != i]
            remaining_nodes = unvisited_ids[remaining_indices]
            
            # Start greedy simulation from the candidate node
            temp_current = node
            temp_remaining = list(remaining_nodes)
            temp_res_cost = 0.0
            
            # Simulate until 1 node remains in temp_remaining
            while len(temp_remaining) > 1:
                # Get distances from temp_current to all remaining nodes
                dists = distance_matrix[temp_current, temp_remaining]
                min_idx = np.argmin(dists)
                next_node_residual = temp_remaining[min_idx]
                dist = dists[min_idx]
                
                temp_res_cost += dist
                temp_current = next_node_residual
                # Remove the visited node from remaining
                # Optimization: swap and pop for O(1) removal if needed, 
                # but list pop is O(N) which is acceptable for small N in look-ahead
                temp_remaining.pop(min_idx)
                
            # Add step to the last remaining node
            last_node = temp_remaining[0]
            temp_res_cost += distance_matrix[temp_current, last_node]
            
            # Add step to destination
            final_step = distance_matrix[last_node, destination_node]
            temp_res_cost += final_step
            
            lookahead_cost = weight_lookahead * temp_res_cost

        # 4. Destination Bias
        # Penalty for being far from the destination, encouraging routes that stay "on track"
        dist_node_to_dest = distance_matrix[node, destination_node]
        dest_bias = weight_dest_bias * dist_node_to_dest
        
        # Composite Score
        score = immediate_dist + stability_penalty + lookahead_cost + dest_bias
        
        if score < best_score:
            best_score = score
            best_node = node
            
    return int(best_node)

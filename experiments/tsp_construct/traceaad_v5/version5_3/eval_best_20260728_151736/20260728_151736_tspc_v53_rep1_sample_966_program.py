import numpy as np
import numpy as np
import math
def _prim_mst_weight(nodes, distance_matrix):
    """
    Compute the weight of the MST for the given nodes using Prim's algorithm.
    nodes: array-like of node IDs
    distance_matrix: full distance matrix
    """
    if len(nodes) <= 1:
        return 0.0
    
    n = len(nodes)
    # Map node IDs to indices 0..n-1 for easier handling
    # We assume nodes is an array of IDs present in distance_matrix
    
    visited = [False] * n
    min_dists = np.full(n, np.inf)
    
    # Start from node 0
    min_dists[0] = 0.0
    
    total_weight = 0.0
    
    for _ in range(n):
        # Find the unvisited node with the smallest distance
        u = -1
        min_val = np.inf
        for i in range(n):
            if not visited[i] and min_dists[i] < min_val:
                min_val = min_dists[i]
                u = i
        
        if u == -1:
            break 
            
        visited[u] = True
        total_weight += min_val
        
        # Update distances to neighbors
        u_node_id = nodes[u]
        for i in range(n):
            if not visited[i]:
                v_node_id = nodes[i]
                dist = distance_matrix[u_node_id, v_node_id]
                if dist < min_dists[i]:
                    min_dists[i] = dist
                    
    return total_weight


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
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_candidates = len(unvisited_nodes)
    total_nodes = distance_matrix.shape[0]
    
    # Precompute distances for efficiency
    dist_current_to_candidates = distance_matrix[current_node, unvisited_nodes]
    dist_candidates_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Calculate average distance among unvisited nodes to normalize immediate costs
    if n_candidates > 1:
        sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
        avg_dist = np.mean(sub_matrix)
        if avg_dist == 0:
            avg_dist = 1.0 # Avoid division by zero
    else:
        avg_dist = 1.0

    # Progress is 0 at start, 1 at end (just before closing)
    progress = (total_nodes - n_candidates) / max(1, total_nodes)
    
    best_score = float('inf')
    best_candidate = unvisited_nodes[0]
    
    # Store estimated costs for all candidates to compute regret
    estimated_costs = []
    candidates_list = []

    # Precompute distances between all unvisited nodes for NN and centrality
    dist_matrix_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]

    # For each candidate, estimate the cost of the partial tour: 
    # current -> candidate -> (nearest unvisited of remaining) ... -> dest
    for idx, candidate in enumerate(unvisited_nodes):
        # Remaining unvisited nodes after picking candidate
        remaining_indices = np.delete(np.arange(n_candidates), idx)
        remaining_nodes = unvisited_nodes[remaining_indices]
        
        # Cost to reach candidate from current node
        cost_to_candidate = dist_current_to_candidates[idx]
        
        # Estimate cost to visit all remaining nodes and then go to destination
        # Use nearest-neighbor heuristic starting from candidate
        estimated_remaining_cost = 0.0
        
        if len(remaining_nodes) > 0:
            # Start NN from the candidate
            temp_pos = candidate
            # We need to visit all remaining_nodes using NN
            # Create a list to track visited order for simulation
            temp_remaining_nodes = list(remaining_nodes)
            
            while temp_remaining_nodes:
                # Find nearest unvisited node from temp_pos
                dists_to_remaining = distance_matrix[temp_pos, temp_remaining_nodes]
                min_idx_in_temp = np.argmin(dists_to_remaining)
                next_node = temp_remaining_nodes[min_idx_in_temp]
                
                # Add distance
                estimated_remaining_cost += dists_to_remaining[min_idx_in_temp]
                
                # Move to next node
                temp_pos = next_node
                
                # Remove visited node from list
                temp_remaining_nodes.pop(min_idx_in_temp)
            
            # Add distance from last visited node to destination
            estimated_remaining_cost += distance_matrix[temp_pos, destination_node]
        else:
            # If no remaining nodes, just go to destination
            estimated_remaining_cost = dist_candidates_to_dest[idx]
        
        # Calculate MST disruption penalty
        mst_remaining = _prim_mst_weight(remaining_nodes, distance_matrix)
        
        n_remaining = len(remaining_nodes)
        mst_penalty = 0.0
        if n_remaining > 0:
            mst_penalty = mst_remaining * progress
            
        total_estimated_cost = cost_to_candidate + estimated_remaining_cost + mst_penalty
        estimated_costs.append(total_estimated_cost)
        candidates_list.append(candidate)
    
    # Find the minimum estimated cost among all candidates
    min_estimated_cost = min(estimated_costs)

    # Calculate Regret Volatility Weight: Coefficient of Variation
    est_costs_arr = np.array(estimated_costs)
    mean_est_cost = np.mean(est_costs_arr)
    std_est_cost = np.std(est_costs_arr)
    
    if mean_est_cost > 0:
        cv = std_est_cost / mean_est_cost
    else:
        cv = 0.0
        
    # Base regret weight
    regret_weight = 1.0 + cv

    # Calculate regret and final score for each candidate
    for idx, candidate in enumerate(unvisited_nodes):
        estimated_cost = estimated_costs[idx]
        immediate_cost = dist_current_to_candidates[idx]
        regret = estimated_cost - min_estimated_cost
        
        # Normalize immediate cost by average distance to prevent scale dominance
        normalized_immediate_cost = immediate_cost / avg_dist
        
        normalized_regret = regret / avg_dist
        
        # Calculate Hub-Centrality Bias
        remaining_indices = np.delete(np.arange(n_candidates), idx)
        if len(remaining_indices) > 0 and n_candidates > 1:
            dists_to_remaining_from_candidate = dist_matrix_unvisited[idx, remaining_indices]
            hub_centrality_bias = np.mean(dists_to_remaining_from_candidate)
        else:
            hub_centrality_bias = 0.0
            
        # Dynamic decay factor for hub-centrality bias
        decay_factor = 1.0 / (1.0 + progress)
        
        # Exponential strandedness penalty
        base_strandedness = math.exp(-progress * n_candidates) * (dist_candidates_to_dest[idx] / avg_dist)
        
        # Apply CV Modulation
        strandedness_modulator = 1.0 / (1.0 + cv)
        strandedness_penalty = base_strandedness * strandedness_modulator
        
        # Lookahead Consistency Modifier
        k_neighbors = min(3, n_candidates - 1)
        consistency_penalty = 0.0
        if k_neighbors > 0:
            sorted_indices = np.argsort(dist_current_to_candidates)
            start_idx = 1
            end_idx = min(k_neighbors + 1, n_candidates)
            
            if end_idx > start_idx:
                neighbor_indices = sorted_indices[start_idx:end_idx]
                neighbor_costs = [estimated_costs[i] for i in neighbor_indices]
                if len(neighbor_costs) > 0:
                    local_mean_cost = np.mean(neighbor_costs)
                    deviation = abs(estimated_cost - local_mean_cost) / avg_dist
                    consistency_penalty = deviation * 0.5
            else:
                consistency_penalty = 0.0

        # Edge Strain Penalty
        # Ratio of immediate distance to estimated remaining cost
        # Scaled by inverse of Hub-Centrality Bias
        # Subtracted from score (lower score is better)
        # Reward efficient transitions through central nodes
        edge_strain = 0.0
        if estimated_cost > 0 and hub_centrality_bias > 0:
            # Add small epsilon to avoid division by zero
            inv_hcb = 1.0 / (hub_centrality_bias + 1e-9)
            ratio = immediate_cost / estimated_cost
            edge_strain = ratio * inv_hcb
        
        # Combine terms
        # Subtracted edge_strain rewards low ratio (efficient step) and low centrality bias (central node)
        total_score = normalized_immediate_cost + regret_weight * normalized_regret - decay_factor * hub_centrality_bias + strandedness_penalty + consistency_penalty - edge_strain
        
        if total_score < best_score:
            best_score = total_score
            best_candidate = candidate
            
    return int(best_candidate)

import numpy as np
from itertools import permutations

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
        # Should not happen in typical usage, but return current or raise error
        raise ValueError("No unvisited nodes left")
    
    if len(unvisited_nodes) == 1:
        # Only one node left to visit before returning to destination
        return unvisited_nodes[0]
    
    # Tiered Strategy: Early exit for small remaining sets (<=3)
    # Use exhaustive permutation search for high precision and low overhead
    if len(unvisited_nodes) <= 3:
        best_cost = float('inf')
        best_next = unvisited_nodes[0]
        
        # Iterate through all permutations of unvisited nodes to find the optimal suffix
        for perm in permutations(unvisited_nodes):
            # Calculate cost of the path: current -> perm[0] -> ... -> perm[-1] -> destination
            temp_cost = distance_matrix[current_node, perm[0]]
            
            # Sum distances within the permutation
            for i in range(len(perm) - 1):
                temp_cost += distance_matrix[perm[i], perm[i+1]]
            
            # Add return edge to destination
            temp_cost += distance_matrix[perm[-1], destination_node]
            
            if temp_cost < best_cost:
                best_cost = temp_cost
                best_next = perm[0]
        
        return best_next

    # Adaptive k-selection based on instance size (from reference trajectory)
    # Scale k with sqrt(N) + 2 to balance exploration in large instances
    n = len(unvisited_nodes)
    k = max(2, min(int(np.ceil(np.sqrt(n))) + 2, n))
    
    # Get distances from current node to all unvisited nodes
    distances_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # Get indices of the k smallest distances
    k_best_indices = np.argsort(distances_to_unvisited)[:k]
    k_best_candidates = unvisited_nodes[k_best_indices]
    
    best_cost = float('inf')
    best_next = k_best_candidates[0]
    
    for candidate in k_best_candidates:
        # Cost to move from current node to candidate
        move_cost = distance_matrix[current_node, candidate]
        
        # Remaining unvisited nodes after visiting candidate
        # Use boolean masking for efficient removal to avoid array reallocation overhead
        remaining = unvisited_nodes[unvisited_nodes != candidate]
        
        if len(remaining) == 0:
            # After visiting candidate, return directly to destination
            total_cost = move_cost + distance_matrix[candidate, destination_node]
        else:
            # Step 1: Construct initial NN tour for the remaining nodes starting from candidate
            nn_tour = [candidate]
            sim_current = candidate
            temp_remaining = remaining.copy()
            
            while len(temp_remaining) > 0:
                # Vectorized distance calculation from sim_current to all temp_remaining
                dists = distance_matrix[sim_current, temp_remaining]
                
                # Find the nearest node using argmin
                nearest_idx = np.argmin(dists)
                nearest_node = temp_remaining[nearest_idx]
                nn_tour.append(nearest_node)
                
                # Update current and remove visited node efficiently using boolean masking
                sim_current = nearest_node
                temp_remaining = temp_remaining[temp_remaining != nearest_node]
            
            # Step 2: Apply iterative 'Reverse-Swap' optimization with Adaptive Terminal Bias
            # This includes standard 2-opt (internal edge swaps) and Suffix Reversals (terminal swaps)
            optimized_tour = nn_tour.copy()
            improved = True
            
            while improved:
                improved = False
                path_len = len(optimized_tour)
                
                # Current endpoint for bias calculation
                current_last_node = optimized_tour[-1]
                current_dist_to_dest = distance_matrix[current_last_node, destination_node]
                # Avoid division by zero if destination is already visited (shouldn't happen) or distance is 0
                base_bias = 1.0 / (current_dist_to_dest + 1e-9)
                
                best_weighted_improvement = -1.0
                best_move_type = None # 'swap' or 'reverse'
                best_i, best_j = -1, -1
                
                # Check Standard 2-opt swaps (internal edges only)
                # Swaps edges (i, i+1) and (j, j+1) with (i, j) and (i+1, j+1)
                # This preserves the endpoints of the tour segment, so last_node doesn't change.
                # Thus, return cost is unchanged. Weighting factor is constant (base_bias).
                for i in range(path_len - 2):
                    for j in range(i + 1, path_len - 1):
                        u1 = optimized_tour[i]
                        v1 = optimized_tour[i+1]
                        u2 = optimized_tour[j]
                        v2 = optimized_tour[j+1]
                        
                        # Current cost of these two edges
                        current_dist = distance_matrix[u1, v1] + distance_matrix[u2, v2]
                        
                        # New edges after reversing segment v1...u2
                        new_dist = distance_matrix[u1, u2] + distance_matrix[v2, v1]
                        
                        path_improvement = current_dist - new_dist
                        
                        # Weighted improvement is just the raw improvement scaled by current bias
                        weighted_improvement = path_improvement * base_bias
                        
                        if weighted_improvement > best_weighted_improvement:
                            best_weighted_improvement = weighted_improvement
                            best_move_type = 'swap'
                            best_i, best_j = i, j
                
                # Check Reverse-Swap moves (Suffix Reversals)
                # Reversing the segment from index i to the end.
                # This changes the edge (i-1, i) and the return edge (last_node, dest).
                # The new last node becomes optimized_tour[i].
                for i in range(1, path_len): # Start from 1 because reversing from 0 is just reversing whole tour
                    # Current edges involved:
                    # 1. Edge from previous node to current start of suffix: optimized_tour[i-1] -> optimized_tour[i]
                    # 2. Return edge: optimized_tour[-1] -> destination_node
                    
                    prev_node = optimized_tour[i-1]
                    current_start_suffix = optimized_tour[i]
                    current_end_suffix = optimized_tour[-1]
                    
                    old_internal_edge = distance_matrix[prev_node, current_start_suffix]
                    old_return_edge = distance_matrix[current_end_suffix, destination_node]
                    
                    old_total_edge_cost = old_internal_edge + old_return_edge
                    
                    # New edges after reversing suffix starting at i:
                    # 1. Edge from previous node to new start of suffix (which is old end): prev_node -> optimized_tour[-1]
                    # 2. Return edge: new last node (which is old start of suffix) -> destination_node
                    
                    new_internal_edge = distance_matrix[prev_node, current_end_suffix]
                    new_return_edge = distance_matrix[current_start_suffix, destination_node]
                    
                    new_total_edge_cost = new_internal_edge + new_return_edge
                    
                    total_improvement = old_total_edge_cost - new_total_edge_cost
                    
                    # Adaptive Terminal Bias:
                    # The new endpoint is current_start_suffix.
                    # We weight the improvement by the inverse of the NEW return distance.
                    # This encourages moves that land closer to destination.
                    new_dist_to_dest = new_return_edge
                    new_bias = 1.0 / (new_dist_to_dest + 1e-9)
                    
                    weighted_improvement = total_improvement * new_bias
                    
                    if weighted_improvement > best_weighted_improvement:
                        best_weighted_improvement = weighted_improvement
                        best_move_type = 'reverse'
                        best_i = i
                        best_j = -1 # Not used for reverse
                
                # Apply best move if improvement found
                if best_weighted_improvement > 0:
                    improved = True
                    if best_move_type == 'swap':
                        i, j = best_i, best_j
                        # Perform standard 2-opt reversal between i+1 and j
                        start_idx = i + 1
                        end_idx = j + 1
                        optimized_tour[start_idx:end_idx] = reversed(optimized_tour[start_idx:end_idx])
                    elif best_move_type == 'reverse':
                        i = best_i
                        # Perform suffix reversal from i to end
                        optimized_tour[i:] = reversed(optimized_tour[i:])

            # Calculate cost of the optimized sub-tour
            sim_cost = 0.0
            for idx in range(len(optimized_tour) - 1):
                sim_cost += distance_matrix[optimized_tour[idx], optimized_tour[idx+1]]
            
            # Add return edge from last node in optimized tour to destination
            last_node = optimized_tour[-1]
            sim_cost += distance_matrix[last_node, destination_node]
            
            total_cost = move_cost + sim_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_next = candidate
    
    return best_next

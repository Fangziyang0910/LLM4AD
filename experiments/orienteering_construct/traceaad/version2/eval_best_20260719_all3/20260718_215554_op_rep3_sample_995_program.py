import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray, prizes: np.ndarray, remaining_budget: float) -> int:
    """
    Design a novel constructive heuristic for the Orienteering Problem.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the route destination node.
    unvisited_nodes: Array of feasible unvisited node IDs. Visiting one of these nodes still leaves enough budget to return to the destination.
    distance_matrix: Pairwise Euclidean distance matrix of all nodes.
    prizes: Prize values of all nodes. The depot prize is 0.
    remaining_budget: Remaining travel budget before selecting the next node.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 0:
        return -1
    
    best_node = -1
    best_score = -float('inf')
    
    # Precompute distances from current node for efficiency
    dist_from_current = distance_matrix[current_node]
    dist_current_to_dest = distance_matrix[current_node][destination_node]
    
    # Identify all feasible nodes from the current position
    # A node is feasible if we can visit it and still return to the destination
    feasible_nodes = []
    for u in unvisited_nodes:
        d = dist_from_current[u]
        d_u_dest = distance_matrix[u][destination_node]
        if d + d_u_dest <= remaining_budget:
            feasible_nodes.append(u)
    
    if not feasible_nodes:
        # If no nodes are feasible to visit and return, we can't visit any. 
        return -1

    # Fixed alpha constant for outer score calculation
    alpha = 1.0
    # Gamma constant for dispersion-adjusted efficiency in simulation
    gamma = 0.5

    for candidate in feasible_nodes:
        # Cost to go from current node to candidate
        cost_to_candidate = dist_from_current[candidate]
        dist_candidate_to_dest = distance_matrix[candidate][destination_node]
        
        # Calculate Detour Cost
        # Detour = (current->candidate->dest) - (current->dest)
        detour_cost = (cost_to_candidate + dist_candidate_to_dest) - dist_current_to_dest
        
        # Remaining budget after visiting candidate
        budget_after_candidate = remaining_budget - cost_to_candidate
        
        # Simulate a greedy completion from candidate to estimate total prize
        # We have visited: ... -> current_node -> candidate
        remaining_unvisited = [u for u in unvisited_nodes if u != candidate]
        
        if len(remaining_unvisited) == 0:
            estimated_prize = prizes[candidate]
        else:
            estimated_prize = prizes[candidate]
            current_sim = candidate
            remaining_budget_sim = budget_after_candidate
            
            # Greedy completion simulation
            # We iterate over a copy or manage indices to avoid modification issues
            temp_unvisited = list(remaining_unvisited)
            
            while temp_unvisited and remaining_budget_sim > 0:
                best_next = -1
                best_ratio = -float('inf')
                
                for u in temp_unvisited:
                    d = distance_matrix[current_sim][u]
                    # Check if we can go to u and then return to destination
                    d_u_dest = distance_matrix[u][destination_node]
                    if d + d_u_dest > remaining_budget_sim:
                        continue
                    
                    # Calculate Dispersion-Adjusted Efficiency
                    # Metric: prize[u] / (distance(current_sim, u) + gamma * distance(u, nearest_unvisited_neighbor_of_u))
                    
                    # Find nearest unvisited neighbor of u (excluding u itself)
                    min_dist_to_neighbor = float('inf')
                    # If temp_unvisited has other nodes besides u
                    other_unvisited = [v for v in temp_unvisited if v != u]
                    if other_unvisited:
                        for v in other_unvisited:
                            d_u_v = distance_matrix[u][v]
                            if d_u_v < min_dist_to_neighbor:
                                min_dist_to_neighbor = d_u_v
                    else:
                        # If u is the only one left, no neighbor benefit
                        min_dist_to_neighbor = 0

                    denominator = d + gamma * min_dist_to_neighbor
                    
                    if denominator > 0:
                        ratio = prizes[u] / denominator
                    else:
                        # Should not happen if d > 0 or gamma*min_dist > 0
                        # If d=0 and min_dist=0 (u is current_sim and isolated?), handle carefully
                        ratio = float('inf') 
                        
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_next = u
                
                if best_next == -1:
                    break
                
                # Visit best_next
                d = distance_matrix[current_sim][best_next]
                estimated_prize += prizes[best_next]
                remaining_budget_sim -= d
                current_sim = best_next
                # Remove best_next from temp_unvisited
                try:
                    temp_unvisited.remove(best_next)
                except ValueError:
                    pass # Should not happen
            
            # End simulation
        
        # Calculate Score
        # Score = Estimated Prize - (alpha * Detour Cost)
        score = estimated_prize - alpha * detour_cost
        
        if score > best_score:
            best_score = score
            best_node = candidate
    
    return best_node

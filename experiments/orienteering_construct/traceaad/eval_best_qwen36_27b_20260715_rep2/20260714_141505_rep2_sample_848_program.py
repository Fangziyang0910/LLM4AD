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
        return destination_node
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Precompute distances from current_node and to destination_node
    dist_current = distance_matrix[current_node]
    dist_dest = distance_matrix[destination_node]
    
    best_node = None
    best_score = -1.0
    
    # Convert unvisited to list for easier manipulation
    unvisited_list = list(unvisited_nodes)
    
    # Recursive helper to estimate future prize with limited depth and return-cost penalty
    def estimate_future_prize(curr_node, budget, avail_nodes, depth):
        if budget <= 0 or len(avail_nodes) == 0 or depth <= 0:
            # Base case: Use simple greedy lookahead for the rest of the path
            sim_curr = curr_node
            sim_bud = budget
            sim_avail = list(avail_nodes)
            total_prize = 0.0
            
            while sim_bud > 0 and len(sim_avail) > 0:
                best_next = None
                best_ratio = -1.0
                best_dist = float('inf')
                
                for u in sim_avail:
                    d_u = distance_matrix[sim_curr][u]
                    d_u_to_dest = dist_dest[u]
                    
                    if d_u + d_u_to_dest <= sim_bud:
                        prize_u = prizes[u]
                        if d_u > 1e-9:
                            ratio = prize_u / d_u
                        else:
                            ratio = float('inf') if prize_u > 0 else 0.0
                        
                        if ratio > best_ratio or (ratio == best_ratio and d_u < best_dist):
                            best_ratio = ratio
                            best_next = u
                            best_dist = d_u
                
                if best_next is None:
                    break
                
                total_prize += prizes[best_next]
                sim_bud -= distance_matrix[sim_curr][best_next]
                sim_curr = best_next
                sim_avail.remove(best_next)
                
            return total_prize

        # Recursive step: Try top K candidates based on greedy ratio and recurse deeper
        feasible_candidates = []
        for u in avail_nodes:
            d_u = distance_matrix[curr_node][u]
            d_u_to_dest = dist_dest[u]
            
            if d_u + d_u_to_dest <= budget:
                prize_u = prizes[u]
                if d_u > 1e-9:
                    ratio = prize_u / d_u
                else:
                    ratio = float('inf') if prize_u > 0 else 0.0
                feasible_candidates.append((u, ratio, d_u, d_u_to_dest))
        
        if not feasible_candidates:
            return 0.0
        
        # Sort by ratio descending to get top candidates
        feasible_candidates.sort(key=lambda x: (-x[1], x[2]))
        
        # Limit the branching factor to keep complexity manageable
        top_k = min(3, len(feasible_candidates))
        
        max_estimated_prize = 0.0
        
        for i in range(top_k):
            candidate_node, _, dist_to_candidate, dist_to_dest = feasible_candidates[i]
            
            # New state after visiting candidate_node
            new_budget = budget - dist_to_candidate
            
            # Calculate tightness coefficient
            if new_budget > 1e-9:
                tightness = dist_to_candidate / new_budget
            else:
                tightness = float('inf')
                
            # Penalty: Distance to destination * Tightness
            penalty = dist_to_dest * tightness
            
            # Immediate score: Prize - Penalty
            immediate_score = prizes[candidate_node] - penalty
            
            new_avail = [u for u in avail_nodes if u != candidate_node]
            
            # Recursive call to estimate future prize from this point
            future_prize = estimate_future_prize(candidate_node, new_budget, new_avail, depth - 1)
            
            # Total estimated prize for this branch
            total_estimated = immediate_score + future_prize
            
            if total_estimated > max_estimated_prize:
                max_estimated_prize = total_estimated
                
        return max_estimated_prize

    # --- K-Nearest Neighbor Filtering Step ---
    # Select K nearest unvisited nodes to current_node
    # Define K (can be tuned, e.g., 10 or 20)
    K = 10
    
    # List to hold candidates with their distance from current node
    distance_candidates = []
    for v in unvisited_list:
        d_v = dist_current[v]
        d_v_to_dest = dist_dest[v]
        
        # Check feasibility: cost to visit + cost to return <= remaining budget
        if d_v + d_v_to_dest <= remaining_budget:
            distance_candidates.append((v, d_v))
            
    # Sort by distance ascending
    distance_candidates.sort(key=lambda x: x[1])
    
    # Keep top K candidates
    candidate_nodes = [item[0] for item in distance_candidates[:K]]
    
    # --- Scoring Step on Reduced Set ---
    # Apply the hybrid scoring (nn_rank + recursive lookahead) exclusively on this reduced set
    
    for v in candidate_nodes:
        d_v = dist_current[v]
        d_v_to_dest = dist_dest[v]
        
        # Check feasibility again (should be true due to filtering, but safe to keep)
        if d_v + d_v_to_dest > remaining_budget:
            continue
            
        # Calculate nn_rank: Prize / (Cost to Node + Min Cost to Destination)
        # This normalizes the prize by the total trip cost required to visit this node and return.
        total_trip_cost = d_v + d_v_to_dest
        if total_trip_cost > 1e-9:
            nn_rank = prizes[v] / total_trip_cost
        else:
            nn_rank = float('inf') if prizes[v] > 0 else 0.0
            
        # Lookahead to estimate future potential
        sim_budget = remaining_budget - d_v
        sim_available = [u for u in unvisited_list if u != v]
        
        # Use recursive estimator for future path
        estimated_future = estimate_future_prize(v, sim_budget, sim_available, depth=3)
        
        # Combine nn_rank (immediate efficiency) and estimated_future (long-term viability)
        lookahead_weight = 0.5 
        
        final_score = nn_rank + lookahead_weight * estimated_future
        
        if final_score > best_score:
            best_score = final_score
            best_node = v
            
    if best_node is None:
        # If no candidate was feasible or scored well, return destination
        return destination_node
        
    return int(best_node)

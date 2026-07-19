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

    best_node_id = -1
    best_score = -np.inf

    # Estimate initial budget as the current remaining budget + distance from current to destination 
    # This is a rough approximation since we don't have the absolute initial budget passed in.
    # However, a better proxy for "pressure" is simply the remaining_budget itself relative to a typical distance scale.
    # To make it robust without external state, we can normalize by the maximum possible distance in the matrix or just use raw budget.
    # A common technique in these heuristics when initial budget isn't available is to use the current remaining_budget as the denominator for a normalized factor, 
    # but that would always be 1.0 for the first step if we compare remaining_after to remaining_before.
    
    # Let's assume a standard normalization: 
    # We can estimate the "Initial Budget" by looking at the total budget constraint usually provided in OP. 
    # Since it's not passed here, we must infer it or use a relative measure.
    # A robust relative measure is: remaining_budget_after / remaining_budget.
    # If remaining_budget_after is close to remaining_budget (low cost move), discount is high.
    # If remaining_budget_after is small (high cost move), discount is low.
    # This captures the "pressure" of the move itself.
    
    # However, the prompt specifically asks for "remaining_budget_after / initial_budget". 
    # Since we don't have initial_budget, we can approximate it. 
    # A reasonable approximation in constructive heuristics without global state is to use the maximum distance in the problem or the current remaining_budget + distance_to_home as a proxy for "total available resource".
    # Let's use the current remaining_budget as a proxy for the "budget pool" available at this stage. 
    # So, discount = remaining_budget_after / remaining_budget.
    # This means:
    # - If we spend little, remaining_budget_after ~ remaining_budget -> discount ~ 1.0 (optimistic)
    # - If we spend a lot, remaining_budget_after << remaining_budget -> discount ~ 0.0 (conservative)
    
    # This effectively penalizes expensive moves by devaluing their future potential, which is desirable.
    
    for candidate_id in unvisited_nodes:
        # Calculate cost to visit this candidate
        dist_to_candidate = distance_matrix[current_node, candidate_id]
        
        # Calculate cost to return from candidate to destination
        dist_candidate_to_dest = distance_matrix[candidate_id, destination_node]
        
        # Check feasibility (though pre-filtered)
        if dist_to_candidate + dist_candidate_to_dest > remaining_budget:
            continue
            
        # Remaining budget after visiting candidate (before returning home, but for future moves, we care about budget left for traveling to next nodes)
        # The budget consumed is dist_to_candidate. The return cost is reserved for the end.
        # So the budget available for *future* nodes is remaining_budget - dist_to_candidate.
        remaining_budget_after = remaining_budget - dist_to_candidate
        
        # Immediate gain
        immediate_prize = prizes[candidate_id]
        
        # Dynamic Discount Factor based on budget pressure
        # Avoid division by zero
        if remaining_budget <= 1e-9:
            discount_factor = 0.0
        else:
            # Discount is proportional to the fraction of budget remaining after the move
            # This makes the algorithm more conservative when the move consumes a large portion of the budget
            discount_factor = remaining_budget_after / remaining_budget
            
        # Estimate future potential
        future_potential = 0.0
        sim_budget = remaining_budget_after
        sim_current = candidate_id
        
        # Get all other unvisited nodes (excluding the candidate we are currently evaluating)
        remaining_unvisited_list = [n for n in unvisited_nodes if n != candidate_id]
        
        sim_visited_count = 0
        max_sim_steps = len(remaining_unvisited_list) + 2 
        
        temp_unvisited = remaining_unvisited_list[:]
        
        while sim_budget > 1e-9 and len(temp_unvisited) > 0:
            best_sim_node = -1
            best_sim_efficiency = -np.inf
            
            sim_curr = sim_current
            
            for node in temp_unvisited:
                d_to_node = distance_matrix[sim_curr, node]
                d_node_to_dest = distance_matrix[node, destination_node]
                
                if d_to_node + d_node_to_dest <= sim_budget:
                    if d_to_node > 1e-9:
                        eff = prizes[node] / d_to_node
                    else:
                        eff = prizes[node] * 1e9
                        
                    if eff > best_sim_efficiency:
                        best_sim_efficiency = eff
                        best_sim_node = node
            
            if best_sim_node == -1:
                break
            
            future_potential += prizes[best_sim_node]
            
            sim_budget -= distance_matrix[sim_curr, best_sim_node]
            sim_current = best_sim_node
            temp_unvisited.remove(best_sim_node)
            
            sim_visited_count += 1
            if sim_visited_count > max_sim_steps:
                break
        
        # Total score is immediate prize + discounted future potential
        score = immediate_prize + discount_factor * future_potential
        
        if score > best_score:
            best_score = score
            best_node_id = candidate_id

    return best_node_id

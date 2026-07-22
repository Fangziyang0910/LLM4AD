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
        return unvisited_nodes[0]
    
    # Avoid division by zero or invalid state
    if remaining_budget <= 1e-9:
        return destination_node

    best_score = -1.0
    best_node = unvisited_nodes[0]
    
    # Cache distances from destination for quick lookup
    dist_to_dest = distance_matrix[destination_node]

    # Initial budget reference for adaptive weighting in final score
    initial_budget_ref = remaining_budget 

    def calculate_score(node_id: int) -> float:
        """
        Calculates the total expected prize score for a candidate node_id.
        """
        dist_to_node = distance_matrix[current_node][node_id]
        
        # Cost to visit this node
        cost_to_node = dist_to_node
        
        # Budget remaining after visiting this node
        budget_after_visit = remaining_budget - cost_to_node
        
        # Check if we can return to destination from this node
        dist_node_to_dest = distance_matrix[node_id][destination_node]
        
        # If we can't even return to destination, this node is invalid
        if budget_after_visit < dist_node_to_dest - 1e-9:
            return 0.0
        
        # Calculate the node's own prize
        current_prize = prizes[node_id]
        
        # Estimate future potential using prize-density lookahead greedy simulation with dynamic penalty exponent
        remaining_budget_sim = budget_after_visit
        current_sim_node = node_id
        future_prize = 0.0
        
        # Create a set of remaining unvisited nodes excluding the current candidate
        other_unvisited = [n for n in unvisited_nodes if n != node_id]
        
        # Greedy simulation:
        # While we have budget and nodes left
        visited_sim = set()
        visited_sim.add(current_sim_node)
        
        temp_remaining_budget = remaining_budget_sim
        
        # We perform steps of greedy insertion to estimate potential
        for _ in range(len(other_unvisited)):
            if temp_remaining_budget <= 1e-9:
                break
            
            # Find the best unvisited node based on adjusted prize-density among feasible nodes
            best_next_node = -1
            best_adjusted_density = -1.0
            
            # Collect all feasible candidates for the next step
            candidates = []
            for next_node in other_unvisited:
                if next_node in visited_sim:
                    continue
                
                dist_sim = distance_matrix[current_sim_node][next_node]
                dist_next_to_dest = dist_to_dest[next_node]
                
                # Budget remaining if we move to next_node
                budget_after_move = temp_remaining_budget - dist_sim
                
                # Check feasibility: cost to return to dest <= remaining budget after move
                if dist_next_to_dest > budget_after_move + 1e-9:
                    continue
                    
                prize = prizes[next_node]
                
                # Calculate base density
                if dist_sim > 1e-9:
                    base_density = prize / dist_sim
                else:
                    base_density = float('inf')
                
                # --- Novel Modification: Local Tightness Fraction ---
                # Calculate Local Tightness: ratio of return distance to remaining budget
                if budget_after_move > 1e-9:
                    local_tightness = dist_next_to_dest / budget_after_move
                else:
                    local_tightness = 1.0 # Fallback, though feasibility check should prevent this
                
                # Clamp tightness to [0, 1] effectively, though feasible nodes imply <= 1.0
                # If tightness is 1.0, it's very risky.
                local_tightness = min(1.0, max(0.0, local_tightness))
                
                # Dynamic Exponent based on local tightness
                # As budget depletes locally (tightness -> 1), alpha increases.
                alpha = 2.0 * local_tightness
                
                # Calculate Return-Risk Penalty
                # The penalty should increase with tightness.
                # Using tightness^alpha.
                # If tightness=1, penalty=1. If tightness=0.5, alpha=1, penalty=0.5.
                risk_penalty = np.power(local_tightness, alpha)
                
                # Adjust density: High risk reduces the attractiveness
                # We subtract the penalty from the density.
                # To ensure penalty has significant impact, we might scale it, but here we rely on density magnitude.
                # If density is high (e.g., 10.0), a penalty of 1.0 is small. 
                # However, tightness is a ratio. Let's assume the penalty is a multiplicative factor or significant additive.
                # Given the previous structure, additive subtraction is consistent.
                adjusted_density = base_density - risk_penalty
                
                candidates.append((next_node, dist_sim, adjusted_density))
            
            if not candidates:
                break
            
            # Sort by adjusted density descending
            candidates.sort(key=lambda x: -x[2])
            
            # Select the best one
            best_next_node = candidates[0][0]
            min_dist_to_next = candidates[0][1]
            
            # Move to best_next_node
            current_prize_add = prizes[best_next_node]
            future_prize += current_prize_add
            temp_remaining_budget -= min_dist_to_next
            current_sim_node = best_next_node
            visited_sim.add(current_sim_node)
        
        raw_future_potential = current_prize + future_prize
        
        # Adaptive Weighting Factor for the final score calculation
        # Scale inversely with remaining budget fraction relative to initial budget of this step.
        if initial_budget_ref > 1e-9:
            budget_fraction = budget_after_visit / initial_budget_ref
            if budget_fraction < 1e-9:
                weight = 1.0
            else:
                weight = 1.0 / budget_fraction
                
            # Score = Current_Prize + (Future_Prize / Weight)
            total_expected_prize = current_prize + (future_prize / weight)
        else:
            total_expected_prize = raw_future_potential

        return total_expected_prize

    # Evaluate all candidates
    scores = {}
    for node_id in unvisited_nodes:
        score = calculate_score(node_id)
        scores[node_id] = score
        if score > best_score + 1e-9:
            best_score = score
            best_node = node_id
        elif abs(score - best_score) < 1e-9:
            pass

    # Tie-breaker: Select the nearest among the tied candidates
    tied_candidates = []
    for node_id in unvisited_nodes:
        if abs(scores[node_id] - best_score) < 1e-9:
            tied_candidates.append(node_id)
            
    if len(tied_candidates) > 1:
        best_dist = float('inf')
        final_node = tied_candidates[0]
        for node_id in tied_candidates:
            d = distance_matrix[current_node][node_id]
            if d < best_dist:
                best_dist = d
                final_node = node_id
        return final_node

    return best_node

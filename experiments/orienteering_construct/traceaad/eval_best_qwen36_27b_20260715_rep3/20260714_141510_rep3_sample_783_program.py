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
        return -1 # Should not happen if unvisited_nodes is filtered correctly
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # Calculate distance from current node to each unvisited node
    distances_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    prizes_of_unvisited = prizes[unvisited_nodes]
    
    # Calculate distance from each unvisited node to destination
    distances_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Feasibility check: ensure we can still return to destination after visiting the node
    # Cost to visit node i and return to destination: dist(current->i) + dist(i->dest)
    total_costs = distances_to_unvisited + distances_to_dest
    
    # Filter out nodes that exceed budget (though input should be pre-filtered)
    feasible_mask = total_costs <= remaining_budget
    feasible_nodes = unvisited_nodes[feasible_mask]
    feasible_distances = distances_to_unvisited[feasible_mask]
    feasible_prizes = prizes_of_unvisited[feasible_mask]
    
    if len(feasible_nodes) == 0:
        return -1
    
    if len(feasible_nodes) == 1:
        return int(feasible_nodes[0])

    # Step 1: Cluster Reachability Score Calculation for all feasible nodes
    # Score = (Estimated Total Prize from Greedy Chain starting at candidate) / dist(current->candidate)
    
    # Dynamic 'Budget Efficiency' Metric for Adaptive K
    # K is proportional to the ratio of remaining_budget to the median distance from current node to feasible nodes.
    # This adapts more robustly to local node density variations than the mean.
    median_distance_scale = np.median(feasible_distances)
    if median_distance_scale == 0:
        # Fallback if median is 0 (e.g., overlapping nodes), use a small epsilon or mean
        median_distance_scale = 1e-10 if np.max(feasible_distances) == 0 else np.mean(feasible_distances)
        
    # Calculate Prize Density Term
    # Ratio of average prize in the feasible set to the global average prize
    # Global average prize considers all nodes (including depot, which is 0), or just unvisited? 
    # The prompt says "global average prize". Usually, this implies the mean of the `prizes` array passed in.
    global_avg_prize = np.mean(prizes)
    
    if global_avg_prize > 0:
        avg_feasible_prize = np.mean(feasible_prizes)
        prize_density_ratio = avg_feasible_prize / global_avg_prize
    else:
        # If global avg is 0, all prizes are 0. Prize density is 1 (neutral) or 0. 
        # Since prizes are non-negative, if global avg is 0, all are 0.
        # Let's clamp it to 1 to avoid zeroing out K, or handle gracefully.
        prize_density_ratio = 1.0 

    # Calculate Budget Slack Penalty
    # Find the minimum cost required to visit any feasible node and return to dest
    # Total cost for node i: dist(current->i) + dist(i->dest)
    # We have feasible_distances (current->i) and corresponding distances_to_dest
    # We need to map feasible_mask back to get distances_to_dest for feasible nodes
    feasible_dist_to_dest = distances_to_dest[feasible_mask]
    
    min_required_cost = np.min(feasible_distances + feasible_dist_to_dest)
    
    # Slack is remaining_budget - min_required_cost
    # If slack is small, we are constrained. 
    # Penalty factor: if slack < 0 (should not happen if feasible), cap at small positive.
    # We want penalty to be 1 when slack is large, and decrease as slack approaches 0.
    # A simple linear penalty: slack / remaining_budget, clamped between 0.1 and 1.0
    # If remaining_budget is 0, avoid division by zero, but feasible implies budget >= cost >= 0
    
    if remaining_budget > 0:
        slack = remaining_budget - min_required_cost
        slack_ratio = slack / remaining_budget
        # Clamp slack_ratio to [0.1, 1.0] to allow at least some exploration
        budget_slack_penalty = np.clip(slack_ratio, 0.1, 1.0)
    else:
        budget_slack_penalty = 0.1 # Fallback

    # Adaptive K calculation
    # A larger budget relative to distance suggests we can explore more nodes, so K increases.
    # We scale K linearly with budget_efficiency, multiplied by prize_density_ratio and budget_slack_penalty.
    budget_efficiency = remaining_budget / median_distance_scale
    raw_k = 2.0 * budget_efficiency * prize_density_ratio * budget_slack_penalty
    
    # Clamp K between 1 and total feasible nodes
    K = max(1, min(int(raw_k), len(feasible_nodes)))
    
    cluster_scores = np.zeros(len(feasible_nodes))
    
    # For each feasible candidate, perform a lightweight simulation to estimate cluster prize
    for i, cand_node in enumerate(feasible_nodes):
        # Budget available for the chain starting at cand_node:
        # Total Budget - (dist(current->cand) + dist(cand->dest))
        # However, the simulation needs to ensure that at every step, the return trip is possible.
        # A simpler approximation for the score denominator is just dist(current->cand).
        # For the numerator, we simulate a greedy path.
        
        dist_current_to_cand = feasible_distances[i]
        
        # Initial state for simulation
        current_sim_node = cand_node
        current_sim_prize = prizes[cand_node]
        
        # Distance incurred from current_node to cand_node
        dist_incurred = dist_current_to_cand
        
        # Remaining nodes excluding cand_node
        mask_remaining = unvisited_nodes != cand_node
        remaining_unvisited = unvisited_nodes[mask_remaining]
        
        while True:
            if len(remaining_unvisited) == 0:
                break
                
            dists_to_remaining = distance_matrix[current_sim_node, remaining_unvisited]
            dists_remaining_to_dest = distance_matrix[remaining_unvisited, destination_node]
            
            # Total cost to visit next and return to dest from start of current_node
            total_costs_for_next = dist_incurred + dists_to_remaining + dists_remaining_to_dest
            
            feasible_next_mask = total_costs_for_next <= remaining_budget
            
            feasible_next_nodes = remaining_unvisited[feasible_next_mask]
            
            if len(feasible_next_nodes) == 0:
                break
                
            # Greedy selection: Choose the node with highest prize-to-distance ratio
            next_prizes = prizes[feasible_next_nodes]
            next_dists = dists_to_remaining[feasible_next_mask]
            
            if np.any(next_dists == 0):
                next_scores = next_prizes / np.where(next_dists == 0, 1e-10, next_dists)
            else:
                next_scores = next_prizes / next_dists
            
            best_next_idx_local = np.argmax(next_scores)
            next_node = feasible_next_nodes[best_next_idx_local]
            next_dist = next_dists[best_next_idx_local]
            
            # Move to next node
            current_sim_node = next_node
            current_sim_prize += prizes[next_node]
            dist_incurred += next_dist
            
            # Remove next_node from remaining_unvisited
            next_idx_in_remaining = np.where(remaining_unvisited == next_node)[0][0]
            remaining_unvisited = np.concatenate([remaining_unvisited[:next_idx_in_remaining], remaining_unvisited[next_idx_in_remaining+1:]])
            
        # Cluster Score: Total Estimated Prize / Distance from Current Node to Candidate
        if dist_current_to_cand > 0:
            cluster_scores[i] = current_sim_prize / dist_current_to_cand
        else:
            cluster_scores[i] = float('inf') if current_sim_prize > 0 else 0.0

    # Select top K candidates based on cluster_scores
    # np.argsort sorts in ascending order, so we take the last K
    top_k_indices = np.argsort(cluster_scores)[-K:]
    
    candidates_indices = top_k_indices
    candidates_nodes = feasible_nodes[top_k_indices]
    
    # Step 2: Detailed Local Chain Simulation for each of the Top K candidates
    # We use the same simulation logic as above, but now we just pick the best one among these K.
    # This ensures we select the node that actually leads to the best path within the high-potential cluster.
    
    best_candidate_node = -1
    max_total_prize = -1.0
    
    for cand_idx in candidates_indices:
        cand_node = feasible_nodes[cand_idx]
        
        # Initial state for simulation
        current_sim_node = cand_node
        current_sim_prize = prizes[cand_node]
        
        # Distance incurred from current_node to cand_node
        dist_incurred = distance_matrix[current_node, cand_node]
        
        # Remaining nodes excluding cand_node
        mask_remaining = unvisited_nodes != cand_node
        remaining_unvisited = unvisited_nodes[mask_remaining]
        
        while True:
            if len(remaining_unvisited) == 0:
                break
                
            dists_to_remaining = distance_matrix[current_sim_node, remaining_unvisited]
            dists_remaining_to_dest = distance_matrix[remaining_unvisited, destination_node]
            
            total_costs_for_next = dist_incurred + dists_to_remaining + dists_remaining_to_dest
            
            feasible_next_mask = total_costs_for_next <= remaining_budget
            
            feasible_next_nodes = remaining_unvisited[feasible_next_mask]
            
            if len(feasible_next_nodes) == 0:
                break
                
            next_prizes = prizes[feasible_next_nodes]
            next_dists = dists_to_remaining[feasible_next_mask]
            
            if np.any(next_dists == 0):
                next_scores = next_prizes / np.where(next_dists == 0, 1e-10, next_dists)
            else:
                next_scores = next_prizes / next_dists
            
            best_next_idx_local = np.argmax(next_scores)
            next_node = feasible_next_nodes[best_next_idx_local]
            next_dist = next_dists[best_next_idx_local]
            
            current_sim_node = next_node
            current_sim_prize += prizes[next_node]
            dist_incurred += next_dist
            
            next_idx_in_remaining = np.where(remaining_unvisited == next_node)[0][0]
            remaining_unvisited = np.concatenate([remaining_unvisited[:next_idx_in_remaining], remaining_unvisited[next_idx_in_remaining+1:]])
            
        if current_sim_prize > max_total_prize:
            max_total_prize = current_sim_prize
            best_candidate_node = cand_node
            
    if best_candidate_node == -1:
        # Fallback: if something went wrong, pick the first candidate
        if len(candidates_nodes) > 0:
            return int(candidates_nodes[0])
        return -1

    return int(best_candidate_node)

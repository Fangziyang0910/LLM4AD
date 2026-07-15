import random
import math
import scipy
try:
    import torch
except Exception:
    torch = None
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
    INITIAL_BUDGET_ESTIMATE = 100.0 
    
    if len(unvisited_nodes) == 0:
        return destination_node
    
    # Calculate distances from current node to each candidate
    dist_to = distance_matrix[current_node, unvisited_nodes]
    
    # Calculate distances from each candidate to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # Total round-trip cost if we visit a node and then go to destination
    round_trip_cost = dist_to + dist_to_dest
    
    # Filter for strict feasibility based on remaining budget
    feasible_mask = round_trip_cost <= remaining_budget
    
    if not np.any(feasible_mask):
        # Fallback: if no node allows visit + return within budget, go to destination
        return destination_node

    candidates = unvisited_nodes[feasible_mask]
    d_to = dist_to[feasible_mask]
    d_to_dest = dist_to_dest[feasible_mask]
    p = prizes[candidates]
    
    # Epsilon for numerical stability
    epsilon = 1e-9
    safe_budget = np.maximum(remaining_budget, epsilon)
    
    # --- Adaptive Prize Scaling (from entail_28_1) ---
    # Instance-Aware Adaptive Prize Scaling
    # k_p = 0.5 + 1.5 * (B / (B + 100))
    budget_scale_factor = remaining_budget / (remaining_budget + INITIAL_BUDGET_ESTIMATE)
    k_p = 0.5 + 1.5 * budget_scale_factor
    prize_scale = 1.0 + k_p / safe_budget
    
    # --- Intermediate Distance Penalty (from rollout_25_1_0_0) ---
    # k_d = 8.0
    k_d = 8.0
    dist_penalty = 1.0 * (1.0 + k_d / safe_budget)
    
    numerator = p * prize_scale - d_to * dist_penalty
    
    # --- Static Denominator (from rollout_25_1_0_0) ---
    # Return Weight 12.0 (Static)
    return_weight_static = 12.0
    denominator = d_to + return_weight_static * d_to_dest
    
    # Ensure denominator is not zero
    safe_denominator = np.maximum(denominator, epsilon)
    
    # Base Score calculation
    base_scores = numerator / safe_denominator
    
    # --- Efficiency Penalty (from entail_28_1 logic) ---
    # Penalize only low efficiency nodes to avoid traps
    round_trip = d_to + d_to_dest
    safe_round_trip = np.maximum(round_trip, epsilon)
    budget_efficiency = p / safe_round_trip
    
    # Normalize efficiency relative to the best candidate in the feasible set
    max_efficiency = np.max(budget_efficiency)
    if max_efficiency > epsilon:
        norm_efficiency = budget_efficiency / max_efficiency
    else:
        norm_efficiency = np.zeros_like(budget_efficiency)
        
    # Apply penalty for low efficiency candidates
    # If normalized efficiency > 0.1, bonus is 0. Otherwise, penalty scales with (1 - norm_eff)
    efficiency_bonus = np.where(norm_efficiency > 0.1, 0.0, -0.5 * (1.0 - norm_efficiency))
    
    total_scores = base_scores + efficiency_bonus
    
    # --- Dynamic Elite Pruning (from rollout_25_1_0_0) ---
    # If the pool of feasible nodes is very small (<=3), bypass adaptive pruning
    if len(candidates) <= 3:
        # Find the index of the maximum score
        best_local_idx = np.argmax(total_scores)
        
        # Handle potential ties using Proximity-First Logic
        max_score = total_scores[best_local_idx]
        tie_mask = np.abs(total_scores - max_score) < epsilon
        tie_candidates = candidates[tie_mask]
        tie_d_to = d_to[tie_mask]
        tie_prizes = p[tie_mask]
        
        if len(tie_candidates) > 1:
            # Select among ties based on minimum distance first
            min_dist = np.min(tie_d_to)
            dist_tie_mask = np.abs(tie_d_to - min_dist) < epsilon
            
            # If there are multiple nodes with the same min distance, break tie by prize
            if np.sum(dist_tie_mask) > 1:
                final_tie_candidates = tie_candidates[dist_tie_mask]
                final_tie_prizes = tie_prizes[dist_tie_mask]
                best_prize_idx = np.argmax(final_tie_prizes)
                best_node_id = final_tie_candidates[best_prize_idx]
            else:
                min_dist_idx = np.argmin(tie_d_to)
                best_node_id = tie_candidates[min_dist_idx]
        else:
            best_node_id = candidates[best_local_idx]
            
        return int(best_node_id)

    # Adaptive pruning for larger candidate pools with DYNAMIC threshold
    max_score = np.max(total_scores)
    
    # Calculate dynamic threshold based on remaining budget ratio
    budget_ratio = remaining_budget / INITIAL_BUDGET_ESTIMATE
    # Clamp budget ratio to [0, 1] for stability
    budget_ratio = np.clip(budget_ratio, 0.0, 1.0)
    
    dynamic_factor = 0.5 + 0.5 * budget_ratio
    threshold = max_score * dynamic_factor
    
    elite_mask = total_scores >= threshold
    
    # If elite mask filters out too many, keep top 3 to ensure selection is possible
    if np.sum(elite_mask) < 3:
        # Get indices of top 3 scores
        top_indices = np.argsort(total_scores)[-3:]
        # Create mask for top indices
        new_mask = np.zeros_like(elite_mask, dtype=bool)
        new_mask[top_indices] = True
        elite_mask = new_mask
        
    candidates = candidates[elite_mask]
    total_scores = total_scores[elite_mask]
    d_to = d_to[elite_mask]
    p = p[elite_mask] # Update prizes array to match elite candidates for tie-breaking
    
    if len(candidates) == 0:
        return destination_node

    # Find the index of the maximum score
    best_local_idx = np.argmax(total_scores)
    
    # Enhance tie-breaking strategy within the elite set using Proximity-First Logic:
    max_score_elite = total_scores[best_local_idx]
    tie_mask_elite = np.abs(total_scores - max_score_elite) < epsilon
    tie_candidates = candidates[tie_mask_elite]
    tie_d_to = d_to[tie_mask_elite]
    tie_prizes = p[tie_mask_elite]
    
    if len(tie_candidates) > 1:
        # Select among ties based on minimum distance first
        min_dist = np.min(tie_d_to)
        dist_tie_mask = np.abs(tie_d_to - min_dist) < epsilon
        
        # If there are multiple nodes with the same min distance, break tie by prize
        if np.sum(dist_tie_mask) > 1:
            final_tie_candidates = tie_candidates[dist_tie_mask]
            final_tie_prizes = tie_prizes[dist_tie_mask]
            best_prize_idx = np.argmax(final_tie_prizes)
            best_node_id = final_tie_candidates[best_prize_idx]
        else:
            min_dist_idx = np.argmin(tie_d_to)
            best_node_id = tie_candidates[min_dist_idx]
    else:
        best_node_id = candidates[best_local_idx]

    # Return the corresponding node ID
    return int(best_node_id)

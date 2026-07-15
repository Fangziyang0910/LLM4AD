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
    if len(unvisited_nodes) == 0:
        return destination_node

    # 1. Extract relevant data for all unvisited nodes
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    dist_to_destination = distance_matrix[unvisited_nodes, destination_node]
    node_prizes = prizes[unvisited_nodes]
    
    # 2. Epsilon-Tightened Hard Feasibility Mask
    epsilon = 1e-9
    feasibility_cost = dist_from_current + dist_to_destination
    feasible_mask = feasibility_cost <= (remaining_budget - epsilon)
    
    if not np.any(feasible_mask):
        return destination_node

    # Filter arrays to only include feasible nodes for scoring
    feasible_indices = np.where(feasible_mask)[0]
    f_dist_from_current = dist_from_current[feasible_mask]
    f_dist_to_destination = dist_to_destination[feasible_mask]
    f_node_prizes = node_prizes[feasible_mask]
    f_unvisited_nodes = unvisited_nodes[feasible_mask]
    
    # 3. Return-aware Base Score (1-step greedy)
    # Formula: Prize / (Dist(Current->Node) + 0.5 * Dist(Node->Dest))
    denominator = f_dist_from_current + 0.5 * f_dist_to_destination
    safe_denominator = np.where(denominator < epsilon, epsilon, denominator)
    base_score = f_node_prizes / safe_denominator
    
    # 4. Level-3 Cluster Depth Look-Ahead Score
    # Calculates the max prize of a feasible 3-step path: Current -> i -> j -> k -> Dest
    # This captures cluster density and future feasibility better than 1 or 2 steps.
    depth_score = np.zeros(len(f_unvisited_nodes))
    feasible_count = np.zeros(len(f_unvisited_nodes)) # For density awareness
    
    if len(f_unvisited_nodes) > 1:
        K = len(f_unvisited_nodes)
        # Pairwise distances between feasible unvisited nodes
        pairwise_dists = distance_matrix[np.ix_(f_unvisited_nodes, f_unvisited_nodes)]
        
        # Precompute cost from any node m to destination: dist(m, dest)
        # Shape: (K,)
        dist_to_dest_vec = f_dist_to_destination
        
        # Cost matrix for j->k->dest: dist(j, k) + dist(k, dest)
        # Shape (K, K): rows are j, cols are k
        cost_j_k_dest = pairwise_dists + dist_to_dest_vec[np.newaxis, :]
        
        # Loop over i (current step candidate)
        for i in range(K):
            rem_after_i_val = remaining_budget - f_dist_from_current[i]
            
            # 1. Identify feasible successors j from i
            # cost i->j->dest <= rem_after_i_val - epsilon
            # But for 3-step, we need i->j->k->dest.
            # So we need: dist(i, j) + min(dist(j, k) + dist(k, dest)) <= rem_after_i_val
            # Let's iterate j, then find best k.
            
            # Cost for i->j->dest is dist(i,j) + dist(j,dest)
            # We need enough budget left after i->j to visit at least one k and return.
            # Constraint: dist(i, j) + dist(j, k) + dist(k, dest) <= rem_after_i_val
            
            # Vectorized approach for i:
            # dist_i_j = pairwise_dists[i, :]
            # rem_after_j = rem_after_i_val - dist_i_j
            
            # For each j, check if there exists k such that:
            # cost_j_k_dest[j, k] <= rem_after_j[j] - epsilon
            # and k != j, k != i
            
            dist_i_j = pairwise_dists[i, :]
            rem_after_j_vec = rem_after_i_val - dist_i_j
            
            # Mask for valid j: rem_after_j must be positive enough to cover min cost to dest?
            # Actually, just check feasibility of k later.
            # But first, j must be reachable from i: dist(i, j) <= rem_after_i_val - epsilon
            feasible_j_reach_mask = dist_i_j <= (rem_after_i_val - epsilon)
            feasible_j_reach_mask = feasible_j_reach_mask & (np.arange(K) != i) # j != i
            
            if not np.any(feasible_j_reach_mask):
                depth_score[i] = 0
                feasible_count[i] = 0 # No feasible j, so no density
                continue

            # For each feasible j, find best k
            # cost_j_k_dest is (K, K). We need row j.
            # We need to mask out k=j and k=i
            
            # Initialize max prize for each j
            best_k_prize_for_j = np.full(K, -1.0)
            
            # Loop j for clarity and to handle masks correctly
            feasible_j_indices = np.where(feasible_j_reach_mask)[0]
            feasible_count[i] = len(feasible_j_indices)
            
            if len(feasible_j_indices) == 0:
                 depth_score[i] = 0
                 continue

            # Vectorized calculation for all feasible j at once
            # Rows: feasible_j_indices
            j_rows = feasible_j_indices
            
            # Get cost matrix for these j rows
            cost_j_k_dest_sub = cost_j_k_dest[j_rows, :] # Shape (num_j, K)
            rem_after_j_sub = rem_after_j_vec[j_rows] # Shape (num_j,)
            
            # Feasibility for k: cost_j_k_dest_sub <= rem_after_j_sub[:, np.newaxis] - epsilon
            feasible_k_mask = cost_j_k_dest_sub <= (rem_after_j_sub[:, np.newaxis] - epsilon)
            
            # Exclude k == j and k == i
            # k == j: indices where col == row index in full matrix?
            # We need to exclude k such that k_id == j_id
            # Create mask for k != j
            k_not_j = np.ones((len(j_rows), K), dtype=bool)
            for idx, j_idx in enumerate(j_rows):
                k_not_j[idx, j_idx] = False
                
            # Exclude k == i
            k_not_i = np.ones((len(j_rows), K), dtype=bool)
            k_not_i[:, i] = False
            
            feasible_k_mask = feasible_k_mask & k_not_j & k_not_i
            
            # Calculate max prize of k for each j
            # Mask prizes
            masked_prizes = feasible_k_mask * f_node_prizes[np.newaxis, :]
            # Max prize for each j
            max_prize_for_j = np.max(masked_prizes, axis=1)
            
            # If no k found, max_prize is 0 (since masked is 0). 
            # But if all feasible k have prize 0? 
            # We want max prize of any valid k. If no valid k, score is 0.
            # Check if any valid k exists for each j
            has_valid_k = np.any(feasible_k_mask, axis=1)
            max_prize_for_j[~has_valid_k] = 0
            
            # The depth score for i is the max prize of the best k reachable via any feasible j
            # i.e., max over j (prize of best k via j)
            depth_score[i] = np.max(max_prize_for_j) if len(max_prize_for_j) > 0 else 0

    # 5. Adaptive Sensitivity Dynamic Budget-Weighted Heuristic
    dist_current_dest = distance_matrix[current_node, destination_node]
    
    # Scale factor for stability
    max_dist_to_dest = np.max(f_dist_to_destination) if len(f_unvisited_nodes) > 0 else 1.0
    scale = max_dist_to_dest if max_dist_to_dest > 1e-8 else 1.0
    
    # Budget ratio
    estimated_base_cost = dist_current_dest + scale
    if estimated_base_cost > 1e-9:
        budget_ratio = remaining_budget / estimated_base_cost
    else:
        budget_ratio = 1.0
        
    budget_ratio = np.clip(budget_ratio, 0.0, 1.0)
    
    # Density Calculation
    k = 5.0
    p = 1.5 
    norm_density = np.power(feasible_count / (feasible_count + k), p)

    # Adaptive Sensitivity
    gamma = 1.0 + 2.0 * norm_density
    
    # Dynamic Weight Function
    x = gamma * budget_ratio - gamma / 2.0
    x = np.clip(x, -10, 10)
    sigmoid_val = 1.0 / (1.0 + np.exp(-x))
    
    w_forward_base = 0.15 + 0.40 * sigmoid_val
    
    # Linear Additive Modulation
    beta = 0.10
    w_forward = w_forward_base + beta * norm_density
    
    # Cap w_forward
    w_forward = np.clip(w_forward, 0.20, 0.55)
    
    w_base = 1.0 - w_forward
    
    # 6. Normalization
    # Base score normalization
    max_base = max(np.max(base_score), epsilon)
    norm_base = base_score / max_base
        
    # Depth score normalization
    max_prize = max(np.max(f_node_prizes), epsilon)
    max_depth = max(np.max(depth_score), epsilon)
    
    # Normalize depth score by prize scale
    divisor_depth = max(max_depth / max_prize, epsilon)
    norm_depth = depth_score / divisor_depth
        
    # 7. Final Score Calculation
    final_score = w_base * norm_base + w_forward * norm_depth
        
    # 8. Composite Tie-Breaking
    best_val = np.max(final_score)
    best_candidates_mask = final_score >= (best_val - epsilon)
    best_indices_local = np.where(best_candidates_mask)[0]
    
    if len(best_indices_local) == 1:
        best_idx_local = best_indices_local[0]
    else:
        # Composite Tie-Breaker: minimize cost to dest
        costs_of_tied = f_dist_to_destination[best_indices_local] + 0.5 * f_dist_from_current[best_indices_local]
        min_cost_idx = np.argmin(costs_of_tied)
        best_idx_local = best_indices_local[min_cost_idx]
    
    # Map back to original unvisited_nodes index
    original_index = feasible_indices[best_idx_local]
    
    return int(unvisited_nodes[original_index])

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
        raise ValueError("No unvisited nodes available.")
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    current_pos = current_node
    dest_pos = destination_node

    # Extract relevant data arrays for broadcasting
    nodes = unvisited_nodes
    prizes_arr = prizes[nodes]
    dist_from_current = distance_matrix[current_pos, nodes]
    dist_to_dest = distance_matrix[nodes, dest_pos]

    # Core Metric: Symmetric Efficiency
    # Score = Prize / (dist_from_current + dist_to_dest)
    epsilon_div = 1e-9
    total_trip_cost = dist_from_current + dist_to_dest
    
    # Calculate base symmetric efficiency scores
    base_scores = prizes_arr / (total_trip_cost + epsilon_div)

    # Strict Feasibility Masking
    # Ensure that the trip cost is within the remaining budget
    epsilon_feas = 1e-6
    feasibility_mask = total_trip_cost <= remaining_budget + epsilon_feas
    
    # Apply strict masking: infeasible nodes get -inf so they are never selected by argmax
    adjusted_scores = np.where(feasibility_mask, base_scores, -np.inf)

    # Fallback: If no nodes are strictly feasible, pick closest to current for progress
    if not np.any(feasibility_mask):
        best_index = np.argmin(dist_from_current)
        return int(nodes[best_index])

    # Deterministic, instance-specific perturbation derived from node IDs
    # This replaces random noise to break ties and simulate exploration deterministically
    # We use a simple hash-like operation on the node IDs to generate unique perturbations
    # scaled to be small relative to the scores to not disrupt the main metric significantly.
    # Assuming node IDs are integers, we can use a simple pseudo-random-ish mapping.
    # To ensure positivity and small magnitude:
    perturbations = np.array([((int(n) * 2654435761) & 0xFFFFFFFF) % 1000 / 100000.0 for n in nodes])
    final_scores = adjusted_scores + perturbations

    # Find the maximum score among feasible nodes
    best_score = np.max(final_scores)
    
    # Identify nodes with scores within a small absolute tolerance of max score
    tolerance = 1e-9
    close_to_max_mask = (final_scores >= best_score - tolerance) & (final_scores <= best_score + tolerance)
    
    # If no nodes found via tolerance, fallback to argmax
    if not np.any(close_to_max_mask):
        best_index = np.argmax(final_scores)
        return int(nodes[best_index])

    candidates = nodes[close_to_max_mask]
    candidate_scores_mask = close_to_max_mask
    
    if len(candidates) == 0:
        best_index = np.argmax(final_scores)
        return int(nodes[best_index])

    # Extract metrics for candidates for Lexicographic Tie-Breaking
    # Prioritize nodes that maximize the remaining budget post-visit (budget preservation)
    candidate_trip_costs = total_trip_cost[candidate_scores_mask]
    
    # Calculate remaining budget after visit for each candidate
    remaining_after_visit = remaining_budget - candidate_trip_costs
    
    # Select the candidate that leaves the most budget
    max_remaining_budget = np.max(remaining_after_visit)
    budget_mask = remaining_after_visit == max_remaining_budget
    
    # Filter candidates by max remaining budget
    final_candidates = candidates[budget_mask]
    
    # If ties persist in budget, select the node with the minimum dist_from_current (local compactness)
    if len(final_candidates) > 1:
        final_dist_from_current = dist_from_current[budget_mask]
        min_dist = np.min(final_dist_from_current)
        dist_mask = final_dist_from_current == min_dist
        best_node_candidates = final_candidates[dist_mask]
        
        # If still tied, pick the first one (or use ID tie-breaking if needed, but first is deterministic)
        best_node = best_node_candidates[0]
    else:
        best_node = final_candidates[0]

    return int(best_node)

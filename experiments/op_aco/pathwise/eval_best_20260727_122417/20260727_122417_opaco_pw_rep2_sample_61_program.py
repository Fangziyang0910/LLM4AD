import random
import math
import scipy
try:
    import torch
except Exception:
    torch = None
import numpy as np
def heuristics(prize: np.ndarray, distance: np.ndarray, maxlen: float) -> np.ndarray:
    """Return edge desirability values for OP ant colony optimization.

    Args:
        prize: Node prizes with shape (n,). Node 0 is the depot.
        distance: Pairwise Euclidean distances with shape (n, n).
            Diagonal entries are large sentinels so self-loops are unused.
        maxlen: Maximum allowed tour length (return-to-depot constrained).

    Returns:
        An (n, n) edge-prior matrix. Larger values make an edge more likely
        to be sampled. Values at or below zero are treated as 1e-9.
    """
    n = len(prize)
    eps = 1e-9
    
    # 1. Base Attraction: Prize[j] / distance[i, j]^2
    # Use safe distance to avoid division by zero
    dist_safe = np.maximum(distance, eps)
    # prize[j] is the reward for visiting node j. 
    # We broadcast prize along the destination axis (columns).
    base_attraction = prize[np.newaxis, :] / (dist_safe ** 2)
    
    # 2. Unified Feasibility Penalty (from entail_2_1)
    
    # Distance from depot (node 0) to current node i
    dist_from_depot = distance[0, :]  # Shape (n,)
    
    # Distance from node j to depot (node 0)
    dist_to_depot = distance[:, 0]    # Shape (n,)
    
    # Correct dist_to_depot for the depot itself (j=0) to avoid double counting in tour estimate
    # For non-depot nodes, this is the return cost. For depot, it's 0 in the context of 
    # estimating the cost of adding node j to a path ending at i.
    # However, the standard estimation is: dist(0, i) + dist(i, j) + dist(j, 0).
    # If j=0, dist(j,0) is 0 (or distance[i,0] is the return leg). 
    # The estimate is meant to bound the full tour: 0->...->i->j->0.
    # If j=0, tour is 0->...->i->0. Cost = dist(0,i) + dist(i,0).
    # So dist_to_depot_corrected[0] should be 0.
    
    dist_to_depot_corrected = dist_to_depot.copy()
    dist_to_depot_corrected[0] = 0.0
    
    # Estimated tour length for edge (i, j):
    # Cost = dist(0, i) + dist(i, j) + dist(j, 0)
    est_tour_length = (dist_from_depot[:, np.newaxis] + 
                       distance + 
                       dist_to_depot_corrected[np.newaxis, :])
    
    # Slack calculation
    slack = maxlen - est_tour_length
    
    # Sharp penalty: Square the positive slack
    slack_term = np.maximum(slack, eps) ** 2
    
    # Smooth decay term: 1 / (1 + (est_tour_length / maxlen)^2)
    ratio = est_tour_length / (maxlen + eps)
    smooth_decay = 1.0 / (1.0 + ratio ** 2)
    
    # Combined feasibility factor
    feasibility_factor = slack_term * smooth_decay
    
    # 3. Depot Proximity Bias (from rollout_1_1_0_0)
    # Applied to all nodes j, but effectively only matters for j != 0 as prize[0]=0
    # Bias: 1 / (1 + dist(j, 0) / maxlen)
    depot_penalty = 1.0 / (1.0 + dist_to_depot[np.newaxis, :] / maxlen)
    
    # Combine Base Attraction, Feasibility Factor, and Depot Penalty
    heuristic = base_attraction * feasibility_factor * depot_penalty
    
    # 4. Depot Return Override (Column j=0)
    # The general formula yields 0 or near-0 for j=0 because prize[0]=0.
    # We explicitly boost return edges if feasible.
    
    # Calculate specific slack for returning to depot from node i:
    # slack_return[i] = maxlen - (dist(0, i) + dist(i, 0))
    # Note: est_tour_length[:, 0] = dist_from_depot[:, None] + distance[:, 0] + 0
    #      = dist(0, i) + dist(i, 0). This is exactly what we need.
    slack_return = slack[:, 0]
    
    # Feasibility mask for return
    mask_feasible_return = slack_return > 0
    
    # Boost calculation:
    # Proportional to remaining slack (linear) and inverse distance.
    dist_to_depot_safe = np.maximum(dist_to_depot, eps)
    
    # Use scale factor 1.0 as suggested by directive to ensure sufficient attraction
    return_boost = (slack_return / dist_to_depot_safe) * 1.0
    
    # Apply boost only if feasible
    heuristic[:, 0] = np.where(mask_feasible_return, return_boost, eps)
    
    # 5. Stability
    heuristic = np.maximum(heuristic, eps)
    heuristic = np.where(np.isfinite(heuristic), heuristic, eps)
    
    return heuristic

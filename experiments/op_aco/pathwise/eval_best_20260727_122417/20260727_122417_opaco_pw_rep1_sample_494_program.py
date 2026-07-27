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
    n = distance.shape[0]
    
    # 1. Robust Dynamic Alpha Calculation
    # Use median distance for robustness against outlier distances
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    
    if np.any(mask):
        non_diag_dist = distance[mask]
        median_dist = np.median(non_diag_dist)
    else:
        median_dist = 1.0
        
    # Fallbacks to prevent division by zero or invalid alpha
    if median_dist <= 1e-9:
        median_dist = 1.0
        
    # Use median prize to stabilize alpha against outlier high prizes
    median_prize = np.median(prize)
    if median_prize <= 1e-9:
        median_prize = np.mean(prize)
    if median_prize <= 1e-9:
        median_prize = 1e-9
        
    # Dynamic Alpha: 2.0 + 0.5 * (median_prize / median_dist)
    alpha = 2.0 + 0.5 * (median_prize / median_dist)
    
    # 2. Density-Scaled Dynamic Beta Curvature (Range [1.5, 3.0])
    # Retained for broader, more robust instance adaptation.
    if median_dist > 0:
        density_proxy = n / (median_dist ** 2)
    else:
        density_proxy = n
        
    beta = 1.5 + 1.5 * np.tanh(density_proxy * 0.1)
    
    # 3. Base Heuristic: Prize / Distance^Alpha
    # Prepare prize destination matrix for broadcasting (prize[j] for all i->j)
    prize_dest = prize[np.newaxis, :]
    
    # Safety for distance to avoid division by zero in power calculation
    dist_safe = np.maximum(distance, 1e-9)
    
    base_heur = prize_dest / (dist_safe ** alpha)
    
    # 4. Refined Prize-Normalized Budget Decay
    # Factor: exp(-distance / (maxlen * prize))
    # Uses 1e-9 floor to prevent inflation of low-prize nodes more effectively than 1.0 floor.
    safe_maxlen = max(maxlen, 1e-9)
    prize_denom = np.maximum(prize_dest, 1e-9)
    
    budget_factor = np.exp(-distance / (safe_maxlen * prize_denom))
    
    # 5. Smooth Curvature Term (Dynamic Beta [1.5, 3.0])
    # Penalizes high-cost edges near budget limit using dynamic beta.
    # Curvature factor: 1 / (1 + (distance / maxlen)^beta)
    curvature_factor = 1.0 / (1.0 + (distance / safe_maxlen) ** beta)
    
    # 6. Slack Calculation
    # Slack = maxlen - dist[i,j] - dist[j,0]
    # Represents remaining budget after taking edge i->j and returning to depot from j
    dist_to_depot = distance[:, 0] # Shape (n,)
    
    # Broadcast dist_to_depot to match distance shape for subtraction
    # distance is (n, n), dist_to_depot needs to be (n, 1) to broadcast across columns (destinations)
    slack = safe_maxlen - distance - dist_to_depot[:, np.newaxis]
    
    # 7. Aggressive Slack Gate (Scale 0.02)
    # Provides stricter feasibility pressure, reducing probability of selecting moves leading to early termination.
    slack_scale = 0.02 * safe_maxlen
    # Clip exponent to prevent overflow/underflow
    slack_gate = np.exp(np.clip(slack / slack_scale, -40, 40))
    
    # 8. Residual Budget Ratio Reward
    # Rewards edges that leave a higher proportion of the budget remaining.
    # Provides continuous gradient signal for feasible paths.
    residual_ratio = np.maximum(slack, 0.0) / safe_maxlen
    residual_reward = np.sqrt(residual_ratio) + 1e-9

    # 9. Corrected 2-Hop Spatial Lookahead
    # Calculate feasibility of extending from j to k given slack after i->j.
    # Condition: dist[j, k] + dist[k, 0] <= slack[i, j]
    
    cost_return = distance[:, 0]
    # Cost matrix for step j->k->0: shape (n, n) where entry [j, k] is dist[j,k] + dist[k,0]
    cost_jk_return = distance + cost_return[np.newaxis, :]
    
    # Broadcast slack (n, n) to (n, n, 1) and cost (n, n) to (1, n, n)
    # S[i, j] vs C[j, k]
    S_exp = slack[:, :, np.newaxis] # (n, n, 1)
    C_exp = cost_jk_return[np.newaxis, :, :] # (1, n, n)
    
    feasible_mask = C_exp <= S_exp # (n, n, n)
    
    # Count feasible k for each (i, j)
    feasible_count = np.sum(feasible_mask, axis=2) # (n, n)
    
    # Normalize by n
    feasibility_score = feasible_count / n
    
    # Lookahead term: (prize[j] * FeasibilityScore[i, j]) / global_max_prize
    global_max_prize = np.max(prize)
    if global_max_prize <= 1e-9:
        global_max_prize = 1e-9
        
    prize_matrix = prize[np.newaxis, :] # (1, n) -> broadcast to (n, n)
    
    lookahead_term = (prize_matrix * feasibility_score) / global_max_prize
    
    # 10. Strict Feasibility Mask
    strict_feasibility_mask = (slack > 0).astype(float)
    
    # 11. Combine Components
    # H = Base_Heur * Refined_Budget_Decay * Smooth_Curvature(Beta) * 
    #     Aggressive_Slack_Gate(0.02) * Residual_Reward * Corrected_2Hop_Lookahead * Strict_Mask
    heur = (base_heur * budget_factor * curvature_factor * 
            slack_gate * residual_reward * lookahead_term * strict_feasibility_mask)
    
    # Ensure no negative values. ACO sampler treats <=0 as 1e-9.
    heur = np.maximum(heur, 1e-9)
    
    return heur

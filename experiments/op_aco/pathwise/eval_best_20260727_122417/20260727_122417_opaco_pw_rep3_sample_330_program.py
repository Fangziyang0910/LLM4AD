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
    return_to_depot = distance[:, 0]
    round_trip_cost = distance + return_to_depot[:, np.newaxis]
    
    # Compute margin: budget remaining after taking edge (i,j) and returning to depot
    margin = maxlen - round_trip_cost
    
    # Determine feasible edges (margin > 0)
    feasible_mask = margin > 0
    
    # Base heuristic: prize[j] / distance[i,j]^1.5
    # Using 1.5 reduces penalty on long-range edges compared to square
    dist_clipped = np.maximum(distance, 1e-9)
    prize_col = prize[np.newaxis, :]
    base_heuristic = prize_col / np.power(dist_clipped, 1.5)
    
    # Calculate local urgency and adaptive alpha
    # Alpha ranges from 1.2 (low urgency) to 0.5 (high urgency)
    max_len_safe = np.maximum(maxlen, 1e-9)
    urgency = np.clip(round_trip_cost / max_len_safe, 0.0, 1.0)
    alpha = 1.2 - 0.7 * urgency
    
    # Compute margin weight for feasible edges only
    # Use np.where to ensure margin base is positive (1e-9) for infeasible edges
    # This prevents negative bases in power function and reduces gradient noise
    margin_safe = np.where(feasible_mask, margin, 1e-9)
    margin_weight = np.power(margin_safe, alpha)
    
    # Combine base heuristic and margin weight
    combined_heuristic = base_heuristic * margin_weight
    
    # Initialize edge_prior with minimum value 1e-9
    # This handles infeasible edges and ensures no zero/negative values
    edge_prior = np.full_like(distance, 1e-9)
    
    # Assign combined_heuristic only to feasible edges
    # This ensures infeasible edges remain at 1e-9 as per policy
    edge_prior[feasible_mask] = combined_heuristic[feasible_mask]
    
    # Ensure no negative or NaN values (safety net)
    edge_prior = np.nan_to_num(edge_prior, posinf=1e-9, neginf=1e-9, nan=1e-9)
    
    # Final clamp to ensure minimum value
    edge_prior = np.maximum(edge_prior, 1e-9)
    
    # Set diagonal to 1e-9 to match global floor and avoid artificial zeros
    np.fill_diagonal(edge_prior, 1e-9)
    
    return edge_prior

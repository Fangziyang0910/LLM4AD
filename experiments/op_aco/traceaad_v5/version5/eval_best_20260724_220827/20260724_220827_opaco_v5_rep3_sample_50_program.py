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
    
    # 1. Calculate base heuristic: prize[j] / distance[i, j]^3
    # Use broadcasting: prize is (n,), distance is (n, n)
    # prize[np.newaxis, :] becomes (1, n), broadcasts to (n, n)
    # distance ** 3 is (n, n)
    dist_cubed = distance ** 3
    heur = prize[np.newaxis, :] / dist_cubed
    
    # 2. Apply State-Aware Feasibility Mask
    # An edge (i, j) is feasible only if the path i -> j -> 0 fits in the 
    # remaining budget from node i.
    # Remaining budget at i = maxlen - distance[i, 0]
    # Cost of move i -> j and return j -> 0 = distance[i, j] + distance[j, 0]
    # Condition: distance[i, j] + distance[j, 0] <= maxlen - distance[i, 0]
    # Rearranged: distance[i, j] + distance[j, 0] + distance[i, 0] <= maxlen
    
    # distance[i, 0] depends on row index i. Shape (n, 1) for broadcasting against (n, n)
    dist_i_to_depot = distance[:, 0:1]  # Shape (n, 1)
    
    # distance[j, 0] depends on column index j. Shape (1, n) for broadcasting
    dist_j_to_depot = distance[0, :]    # Shape (n,) -> need (1, n)
    dist_j_to_depot_matrix = dist_j_to_depot[np.newaxis, :]
    
    # Total cost consideration: i->j + j->0 + i->0
    # distance is (n, n) representing distance[i, j]
    total_cycle_cost = distance + dist_j_to_depot_matrix + dist_i_to_depot
    
    # Mask: True if the cycle cost is within the global maxlen
    feasible_mask = (total_cycle_cost <= maxlen)
    
    # Apply mask: set infeasible edges to a small value (1e-9)
    # This effectively removes them from consideration in ACO sampling
    heur = np.where(feasible_mask, heur, 1e-9)
    
    # 3. Handle special cases
    # a. Diagonal (self-loops): Should be negligible. 
    np.fill_diagonal(heur, 1e-9)
    
    # b. Ensure no negative values and minimum floor
    # Values at or below zero are treated as 1e-9 per contract.
    heur = np.maximum(heur, 1e-9)
    
    return heur

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
    # Calculate slack: remaining budget after taking edge (i,j) and returning from j to depot (0)
    slack = maxlen - distance - distance[:, 0]
    
    # Calculate quartic efficiency: (prize of destination node / edge cost)^4
    eff_cb = (prize / distance) ** 4
    
    # Calculate direct distance penalty: (1 - distance / maxlen)
    # This penalizes edges that use a large portion of the total budget
    dist_penalty = 1.0 - (distance / maxlen)
    
    # Compute heuristic: eff_cb * (slack/maxlen) * dist_penalty if slack > 0, else 1e-9
    # The dist_penalty is only applied if slack > 0 (feasible edge), otherwise 1e-9
    # Note: dist_penalty can be negative if distance > maxlen, but those edges are infeasible (slack < 0 usually implies distance > maxlen/2 approx, but strictly slack < 0 covers infeasibility)
    # We rely on slack > 0 check. If slack > 0, distance < maxlen - return_distance. 
    # dist_penalty might still be negative if distance > maxlen, but if distance > maxlen, slack is definitely negative.
    # So inside the true branch of where, dist_penalty is likely positive or small negative?
    # Actually if slack > 0, then distance + dist[j,0] < maxlen. Thus distance < maxlen. So dist_penalty > 0.
    
    heuristic = np.where(slack > 0, eff_cb * (slack / maxlen) * dist_penalty, 1e-9)
    
    # Ensure diagonal is set to minimal value to prevent self-loops
    np.fill_diagonal(heuristic, 1e-9)
    
    return heuristic

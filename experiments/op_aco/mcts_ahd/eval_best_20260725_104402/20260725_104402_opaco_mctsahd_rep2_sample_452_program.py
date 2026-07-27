
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
    epsilon = 1e-9
    n = prize.shape[0]
    
    # Distances from each node to the depot (node 0)
    dist_to_depot = distance[:, 0]
    
    # 1. Feasibility Mask
    # An edge (i, j) is only feasible if visiting j and returning to depot fits in maxlen.
    # Cost of segment i->j + return j->0
    # Shape: (n, n)
    cost_segment_return = distance + dist_to_depot[np.newaxis, :]
    feasible_mask = cost_segment_return <= maxlen
    
    # 2. Heuristic Calculation
    
    # Base Efficiency: Square of (Prize[j] / dist(i, j))
    # Squaring emphasizes high prize-to-distance ratios more strongly than linear.
    safe_dist_ij = np.maximum(distance, epsilon)
    base_efficiency = (prize[np.newaxis, :] / safe_dist_ij) ** 2
    
    # Local Dominance Ratio:
    # Normalize base_efficiency by the maximum feasible efficiency from each source node i.
    masked_efficiency = np.where(feasible_mask, base_efficiency, 0.0)
    max_efficiency_per_source = np.max(masked_efficiency, axis=1)  # Shape (n,)
    safe_max_eff = np.maximum(max_efficiency_per_source[:, np.newaxis], epsilon)
    dominance_ratio = base_efficiency / safe_max_eff  # Shape: (n, n)
    
    # Slack Bonus:
    # Instead of an exponential penalty, use a linear bonus based on remaining budget fraction.
    # Remaining Budget Fraction = 1 - (Cost / maxlen)
    # We add 1 + k * remaining_fraction to create a multiplier > 1.
    # Using k=2.0 for stronger incentive to save budget.
    cost_ratio = cost_segment_return / (maxlen + epsilon)
    remaining_fraction = np.maximum(1.0 - cost_ratio, 0.0)
    slack_bonus = 1.0 + 2.0 * remaining_fraction
    
    # 3. Combine Components
    # H = Base Efficiency * Dominance Ratio * Slack Bonus
    H = base_efficiency * dominance_ratio * slack_bonus
    
    # 4. Apply Feasibility Mask
    # Infeasible edges get 0.0
    H = np.where(feasible_mask, H, 0.0)
    
    # 5. Post-processing
    # Ensure all values are positive and finite. 
    # Replace non-positive or non-finite values with epsilon.
    H = np.where((H <= 0) | ~np.isfinite(H), epsilon, H)
    
    return H

import numpy as np
from itertools import permutations

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Selects the next node using a hybrid lookahead, isolation penalty, and regret-based stabilization.
    
    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if unvisited_nodes is None or len(unvisited_nodes) == 0:
        return int(destination_node)
    
    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])
    
    candidates = np.asarray(unvisited_nodes, dtype=int)
    current_node = int(current_node)
    destination_node = int(destination_node)
    
    # Exact endgame for small n: exhaustive permutation of all remaining nodes
    if n <= 6:
        best_cost = float('inf')
        best_first = candidates[0]
        for perm in permutations(candidates):
            cost = distance_matrix[current_node, perm[0]]
            for i in range(len(perm) - 1):
                cost += distance_matrix[perm[i], perm[i + 1]]
            cost += distance_matrix[perm[-1], destination_node]
            if cost < best_cost:
                best_cost = cost
                best_first = perm[0]
        return int(best_first)
    
    # 1. Distances
    dist_current = distance_matrix[current_node, candidates]
    dist_dest = distance_matrix[candidates, destination_node]
    dist_cand_cand = distance_matrix[np.ix_(candidates, candidates)]
    
    # 2. Hybrid Exit Cost (from Algorithm 3)
    # Best 1-step escape to another unvisited node or destination
    cand_exit_no_self = dist_cand_cand.copy()
    np.fill_diagonal(cand_exit_no_self, np.inf)
    
    min_exit_1 = np.minimum(np.min(cand_exit_no_self, axis=1), dist_dest)
    mean_exit_1 = np.mean(cand_exit_no_self, axis=1)
    hybrid_exit_1 = 0.6 * min_exit_1 + 0.4 * mean_exit_1
    
    # 3. Two-Step Future Cost (from Algorithm 3)
    # min over next candidate j of dist(i,j) + hybrid_exit_1(j)
    two_step_future = np.min(cand_exit_no_self + hybrid_exit_1[np.newaxis, :], axis=1)
    
    # 4. Safety/Isolation Cost (from Algorithm 2)
    # Penalizes nodes that are far from the "cluster" of remaining nodes (high min distance to others)
    # We use the min distance to other unvisited nodes, capped by the distance to destination
    # (if a node is close to destination, it's a safe exit point even if far from others)
    min_to_other_unvisited = np.min(cand_exit_no_self, axis=1)
    safety_costs = np.minimum(min_to_other_unvisited, dist_dest)
    
    # 5. Scale Normalization (from Algorithm 3)
    # Normalize costs to stay balanced as the tour shrinks
    mean_cand_dist = np.mean(dist_cand_cand[dist_cand_cand > 0]) if np.any(dist_cand_cand > 0) else 1.0
    scale = max(np.mean(dist_current), mean_cand_dist, 1e-9)
    
    norm_current = dist_current / scale
    norm_future = two_step_future / scale
    norm_safety = safety_costs / scale
    
    # 6. Adaptive Weights
    # As n decreases, prioritize future completion and safety
    # Alpha: weight for future lookahead. Starts lower, increases.
    # Beta: weight for safety/isolation. Starts lower, increases.
    
    # Ramp from 0 (large n) to 1 (small n)
    # For n=7, ramp is small. For n=100, ramp is ~0.1.
    # Let's use a soft ramp that starts becoming significant around n < 20.
    ramp = 1.0 / (1.0 + n / 10.0)
    
    alpha = 0.3 * (1.0 - ramp) + 0.8 * ramp  # 0.3 to 0.8
    beta = 0.1 * (1.0 - ramp) + 0.4 * ramp    # 0.1 to 0.4
    
    # 7. Composite Primary Score
    primary_scores = norm_current + alpha * norm_future + beta * norm_safety
    
    # 8. Regret-Based Stabilization (from Algorithm 2)
    # Find the best and second-best scores
    if n > 2:
        # Get indices of the two smallest values
        part_idx = np.argpartition(primary_scores, 2)[:2]
        s1_idx, s2_idx = part_idx
        s1 = primary_scores[s1_idx]
        s2 = primary_scores[s2_idx]
        # Ensure s1 is the minimum
        if s1 > s2:
            second_val = s1
        else:
            second_val = s2
    else:
        # n == 2 case is handled by permutation, but just in case
        best_idx_tmp = np.argmin(primary_scores)
        other_idx = 1 - best_idx_tmp
        second_val = primary_scores[other_idx]
        
    # Regret = second_best - current_score
    # Higher regret means this option is significantly better.
    # We subtract a small fraction of regret from the score to favor clear winners.
    lambda_regret = 0.1
    composite_scores = primary_scores - lambda_regret * (second_val - primary_scores)
    
    final_idx = np.argmin(composite_scores)
    return int(candidates[final_idx])
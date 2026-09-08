import numpy as np
from itertools import permutations

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Hybrid TSP node selection with exact endgame, multi-step lookahead, safety penalties, and regret stabilization.
    """
    if unvisited_nodes is None or len(unvisited_nodes) == 0:
        return int(destination_node)
    
    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])
    
    candidates = np.asarray(unvisited_nodes, dtype=int)
    current_node = int(current_node)
    destination_node = int(destination_node)
    
    # Exact endgame for small n
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
    
    # Distance vectors
    dist_current = distance_matrix[current_node, candidates]
    dist_dest = distance_matrix[candidates, destination_node]
    dist_cand_cand = distance_matrix[np.ix_(candidates, candidates)]
    
    # Exit costs
    cand_exit = dist_cand_cand.copy()
    np.fill_diagonal(cand_exit, np.inf)
    
    min_exit = np.minimum(np.min(cand_exit, axis=1), dist_dest)
    mean_exit = np.mean(cand_exit, axis=1)
    
    # 2-step lookahead: cost of current->i->(best next from i)
    # Best next from i is min(min_exit[i], but we need to pick a specific node j)
    # Actually, let's compute: for each i, min over j!=i of dist(i,j) + min_exit[j]
    # This approximates the cost if we go i->j and then take the cheapest exit from j
    two_step = np.min(cand_exit + min_exit[np.newaxis, :], axis=1)
    
    # 3-step lookahead: cost of current->i->j->(best exit from j)
    # This is similar to two_step but uses a slightly different aggregation
    # Let's use a blend of min and mean for robustness
    three_step = np.min(cand_exit + (0.7 * min_exit + 0.3 * mean_exit)[np.newaxis, :], axis=1)
    
    future_cost = 0.5 * two_step + 0.5 * three_step
    
    # Safety cost: isolation from other unvisited nodes
    safety_cost = np.min(cand_exit, axis=1)
    
    # Normalize costs
    scale = max(np.mean(dist_current), np.mean(dist_cand_cand), 1e-9)
    norm_current = dist_current / scale
    norm_future = future_cost / scale
    norm_safety = safety_cost / scale
    
    # Adaptive weights
    # As n decreases, prioritize destination completion (future_cost)
    alpha = np.clip(0.6 - (n - 6) / 50.0, 0.3, 0.6)
    beta = np.clip(0.2 - (n - 6) / 100.0, 0.05, 0.2)
    
    primary_scores = norm_current + alpha * norm_future + beta * norm_safety
    
    # Regret stabilization
    if n > 2:
        two_smallest_idx = np.argpartition(primary_scores, 2)[:2]
        s1 = primary_scores[two_smallest_idx[0]]
        s2 = primary_scores[two_smallest_idx[1]]
        second_val = max(s1, s2)
    else:
        second_val = primary_scores[1 - np.argmin(primary_scores)]
    
    regret = second_val - primary_scores
    composite_scores = primary_scores - 0.1 * regret
    
    best_idx = np.argmin(composite_scores)
    return int(candidates[best_idx])
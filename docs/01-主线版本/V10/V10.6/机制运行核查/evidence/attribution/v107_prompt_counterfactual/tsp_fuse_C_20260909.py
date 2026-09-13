import numpy as np
from itertools import permutations

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Select the next node using a hybrid of exact endgame, multi-step lookahead,
    destination closure, adaptive weights, and regret stabilization.
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
                cost += distance_matrix[perm[i], perm[i+1]]
            cost += distance_matrix[perm[-1], destination_node]
            if cost < best_cost:
                best_cost = cost
                best_first = perm[0]
        return int(best_first)
    
    # Distances
    dist_current = distance_matrix[current_node, candidates]
    dist_dest = distance_matrix[candidates, destination_node]
    dist_cand_cand = distance_matrix[np.ix_(candidates, candidates)]
    
    # Candidate-to-candidate without self
    cand_exit = dist_cand_cand.copy()
    np.fill_diagonal(cand_exit, np.inf)
    
    # 1. Hybrid exit cost: blend of min and mean distance to other unvisited or destination
    min_exit_1 = np.minimum(np.min(cand_exit, axis=1), dist_dest)
    mean_exit_1 = np.mean(cand_exit, axis=1)
    hybrid_exit_1 = 0.6 * min_exit_1 + 0.4 * mean_exit_1
    
    # 2. 2-step lookahead: min over j of dist(i,j) + hybrid_exit_1(j)
    two_step = np.min(cand_exit + hybrid_exit_1[np.newaxis, :], axis=1)
    
    # 3. 3-step lookahead: min over j of dist(i,j) + min_exit_1(j)
    three_step = np.min(cand_exit + min_exit_1[np.newaxis, :], axis=1)
    
    # 4. Destination closure: average distance of remaining nodes (excluding i) to destination
    sum_dest = np.sum(dist_dest)
    avg_dest_rem = (sum_dest - dist_dest) / (n - 1)
    
    # 5. Robust scale normalization
    # Use max of mean current distance and mean inter-candidate distance
    scale_candidates = np.mean(dist_cand_cand)
    if scale_candidates < 1e-9:
        scale_candidates = 1.0
    scale = max(np.mean(dist_current), scale_candidates)
    if scale < 1e-9:
        scale = 1.0
    
    norm_current = dist_current / scale
    norm_future = (0.5 * two_step + 0.5 * three_step) / scale
    norm_closure = avg_dest_rem / scale
    
    # 6. Adaptive weights
    # As n decreases from 7, increase emphasis on future and closure
    # alpha: future weight, beta: closure weight
    alpha = 0.3 + 0.5 / (n - 6)
    beta = 0.1 + 0.3 / np.sqrt(n - 6)
    
    # Composite score
    scores = 1.0 * norm_current + alpha * norm_future + beta * norm_closure
    
    # 7. Regret-based stabilization
    sorted_idx = np.argsort(scores)
    min_val = scores[sorted_idx[0]]
    second_min_val = scores[sorted_idx[1]]
    regret = second_min_val - scores
    final_scores = scores - 0.1 * regret
    
    best_idx = np.argmin(final_scores)
    return int(candidates[best_idx])
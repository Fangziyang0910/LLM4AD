import numpy as np
from itertools import permutations

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Select the next node using a composite heuristic with adaptive weights,
    2-step lookahead, destination closure, and regret stabilization.
    Uses exact permutation search for small n (n <= 6).
    """
    if unvisited_nodes is None or len(unvisited_nodes) == 0:
        return int(destination_node)
    
    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])
    
    candidates = np.asarray(unvisited_nodes, dtype=int)
    current_node = int(current_node)
    destination_node = int(destination_node)
    
    # Exact endgame for n <= 6
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
    
    # Heuristic for n > 6
    dist_current = distance_matrix[current_node, candidates]
    dist_dest = distance_matrix[candidates, destination_node]
    dist_cand_cand = distance_matrix[np.ix_(candidates, candidates)]
    
    # Prepare candidate-to-candidate matrix without self-loops
    dist_cand_cand_copy = dist_cand_cand.copy()
    np.fill_diagonal(dist_cand_cand_copy, np.inf)
    
    # 1. Exit cost: min distance to any other unvisited node or destination
    # We consider the destination as a potential "exit" only if it were allowed,
    # but in TSP we must visit all unvisited first. So the "exit" from a candidate
    # to continue the tour is to another unvisited node.
    # However, Algorithm 2/3 blended min to unvisited and min to dest.
    # Let's use min distance to other unvisited nodes for the immediate next step in the tour.
    min_exit_unvisited = np.min(dist_cand_cand_copy, axis=1)
    
    # 2. 2-step lookahead: min over j != i of (dist(i,j) + min_exit_unvisited(j))
    # This estimates the cost of moving from i to j, and then from j to its best next node.
    lookahead_2step = np.min(dist_cand_cand_copy + min_exit_unvisited[np.newaxis, :], axis=1)
    
    # 3. Destination closure: average distance of remaining nodes (excluding candidate i) to destination
    # This term encourages visiting nodes that are closer to the destination early on,
    # or rather, it penalizes leaving nodes that are far from the destination for last.
    # Actually, if we visit a node i, the remaining nodes are unvisited \ {i}.
    # The "closure" cost is the estimated cost to finish the tour from the remaining nodes to dest.
    # A simpler proxy: The average distance of the *remaining* nodes to the destination.
    # If we pick i, the remaining nodes are j != i.
    sum_dist_to_dest = np.sum(dist_dest)
    # Average distance of remaining nodes to destination
    if n > 1:
        closure_cost = (sum_dist_to_dest - dist_dest) / (n - 1)
    else:
        closure_cost = 0.0
    
    # 4. Scale normalization
    # Use global max distance or mean distance for stability
    max_dist = np.max(distance_matrix)
    if max_dist == 0:
        max_dist = 1.0
    
    norm_current = dist_current / max_dist
    norm_lookahead = lookahead_2step / max_dist
    norm_closure = closure_cost / max_dist
    
    # 5. Adaptive weights
    # As n decreases, future structure and closure become more critical.
    # For n=7, weights are moderate; as n grows, we prioritize immediate greedy choice.
    
    # Alpha for lookahead: starts high for small n, decreases for large n
    # Beta for closure: starts high for small n, decreases for large n
    
    # Ramp factor: 1 for n=7, smaller for larger n
    # Using 1/sqrt(n-6) for gradual decay
    ramp = 1.0 / np.sqrt(n - 6)
    
    alpha = 0.4 * ramp  # Weight for 2-step lookahead
    beta = 0.2 * ramp   # Weight for destination closure
    
    # Composite score
    scores = 1.0 * norm_current + alpha * norm_lookahead + beta * norm_closure
    
    # 6. Regret-based stabilization
    # Find the second-best score to calculate regret
    if n > 2:
        sorted_idx = np.argsort(scores)
        min_val = scores[sorted_idx[0]]
        second_min_val = scores[sorted_idx[1]]
    else:
        # n is 7+ here, so n > 2 is always true
        sorted_idx = np.argsort(scores)
        min_val = scores[sorted_idx[0]]
        second_min_val = scores[sorted_idx[1]]
    
    regret = second_min_val - scores
    
    # Minimize score - lambda * regret
    # This favors the unique best candidate
    lambda_regret = 0.1
    final_scores = scores - lambda_regret * regret
    
    best_idx = np.argmin(final_scores)
    return int(candidates[best_idx])
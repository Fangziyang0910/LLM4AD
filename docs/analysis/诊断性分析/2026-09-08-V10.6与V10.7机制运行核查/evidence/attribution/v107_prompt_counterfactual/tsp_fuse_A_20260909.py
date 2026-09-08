import numpy as np
from itertools import permutations

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Select the next node using a hybrid heuristic combining immediate cost, 
    destination lookahead, isolation penalty, and regret-based stabilization.
    """
    if unvisited_nodes is None or len(unvisited_nodes) == 0:
        return int(destination_node)

    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])

    candidates = np.asarray(unvisited_nodes, dtype=int)
    current_node = int(current_node)
    destination_node = int(destination_node)

    # Exhaustive endgame for small n (retained from Algorithm 3)
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

    # Distances
    dist_current = distance_matrix[current_node, candidates]
    dist_dest = distance_matrix[candidates, destination_node]
    
    # Compute candidate-to-candidate distances
    dist_cand_cand = distance_matrix[np.ix_(candidates, candidates)]
    np.fill_diagonal(dist_cand_cand, np.inf)
    
    # Min distance to any other unvisited node
    min_to_other_unvisited = np.min(dist_cand_cand, axis=1)
    
    # Safety cost: min distance to other unvisited nodes or destination
    # This penalizes isolated nodes
    safety_costs = np.minimum(min_to_other_unvisited, dist_dest)
    
    # Lookahead cost: approximate future cost towards destination
    # Use a simple estimate: min distance to destination (since we must eventually return)
    # But to be more forward-looking, consider the 2-step path:
    # Current -> Candidate -> Best Next -> ...
    # For simplicity and speed, use the destination distance as a proxy for future cost,
    # weighted by a factor that decreases as n decreases (we care more about it later)
    lookahead_costs = dist_dest.copy()
    
    # Scale normalization (from Algorithm 3)
    scale_candidates = np.mean(dist_cand_cand) if n > 1 else 1.0
    if scale_candidates < 1e-9:
        scale_candidates = 1.0
    scale = max(np.mean(dist_current), scale_candidates)
    if scale < 1e-9:
        scale = 1.0

    norm_current = dist_current / scale
    norm_safety = safety_costs / scale
    norm_lookahead = lookahead_costs / scale

    # Adaptive weights
    # As n decreases, increase weight on lookahead and safety
    # Start with lower weights for lookahead/safety when n is large
    ramp = (10.0 / n) / (1.0 + 10.0 / n)  # 0 for large n, ~1 for n=2
    w_lookahead = 0.5 * ramp
    w_safety = 0.3 * ramp

    # Composite primary score
    primary_scores = norm_current + w_lookahead * norm_lookahead + w_safety * norm_safety

    # Regret-based stabilization (from Algorithm 2)
    if n > 2:
        two_smallest_indices = np.argpartition(primary_scores, 2)[:2]
        if primary_scores[two_smallest_indices[0]] < primary_scores[two_smallest_indices[1]]:
            second_val = primary_scores[two_smallest_indices[1]]
        else:
            second_val = primary_scores[two_smallest_indices[0]]
    else:
        second_val = primary_scores[1] if primary_scores[0] < primary_scores[1] else primary_scores[0]

    regret = second_val - primary_scores
    composite_scores = primary_scores - 0.05 * regret

    best_idx = np.argmin(composite_scores)
    return int(candidates[best_idx])
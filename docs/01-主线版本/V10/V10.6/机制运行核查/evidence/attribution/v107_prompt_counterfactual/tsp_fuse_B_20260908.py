import numpy as np
from itertools import permutations

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.
    
    Combines exhaustive search for small n, adaptive safety/destination lookahead, 
    and regret-based stabilization for large n.
    """
    if unvisited_nodes is None or len(unvisited_nodes) == 0:
        return int(destination_node)
    
    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])
    
    candidates = np.asarray(unvisited_nodes, dtype=int)
    current_node = int(current_node)
    destination_node = int(destination_node)
    
    # 1. Exact endgame for small n (from Algorithm 3)
    # Exhaustive permutation search ensures optimal choice when the search space is small enough.
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
    
    # 2. Heuristic for larger n (Combining Algorithm 2 and 3 elements)
    
    # Immediate cost: distance from current to candidate
    dist_current = distance_matrix[current_node, candidates]
    
    # Destination cost: distance from candidate to destination
    dist_dest = distance_matrix[candidates, destination_node]
    
    # Lookahead cost (from Algorithm 2): 
    # Estimates the cost of finishing the tour if we pick this node.
    # Approximation: The cheapest node among the remaining ones to connect to destination.
    # If the current candidate is the cheapest, use the second cheapest to avoid double counting.
    sorted_indices = np.argsort(dist_dest)
    min_val_1 = dist_dest[sorted_indices[0]]
    min_val_2 = dist_dest[sorted_indices[1]]
    min_idx_1 = sorted_indices[0]
    
    lookahead_costs = np.full(n, min_val_1)
    lookahead_costs[min_idx_1] = min_val_2
    
    # Safety cost (from Algorithm 2):
    # Penalizes nodes that are isolated from the rest of the unvisited cluster.
    # This prevents creating "bridges" that are too far from the main group.
    cand_to_cand = distance_matrix[np.ix_(candidates, candidates)]
    np.fill_diagonal(cand_to_cand, np.inf)
    
    # Min distance to any other unvisited node
    min_to_other_unvisited = np.min(cand_to_cand, axis=1)
    
    # A node is "safe" if it is close to other unvisited nodes OR close to the destination (potential exit).
    safety_costs = np.minimum(min_to_other_unvisited, dist_dest)
    
    # 3. Adaptive Weights
    # As n decreases, the global structure matters more, so we increase the weight of 
    # destination proximity (lookahead) and connectivity (safety).
    
    # Alpha: weight for lookahead. Increases as n drops.
    # Beta: weight for safety. Also increases slightly.
    
    # Use a soft ramp to transition weights.
    # When n is large (e.g., 100), weights are small. When n is small (e.g., 7), weights are larger.
    ramp = 1.0 / (1.0 + 0.1 * (n - 6))
    
    alpha = 0.4 * ramp
    beta = 0.2 * ramp
    
    # 4. Composite Primary Score
    # We normalize costs to ensure stability across different instance scales?
    # Algorithm 2 did not normalize explicitly, relying on the relative magnitude.
    # However, dist_current is the primary driver. 
    # Let's stick to the additive form from Algorithm 2 which performed well.
    
    primary_scores = dist_current + alpha * lookahead_costs + beta * safety_costs
    
    # 5. Regret-based Stabilization (from Algorithm 2/1)
    # Prefer candidates that are clearly better than the second best.
    if n > 2:
        # Find the two smallest scores
        two_smallest_indices = np.argpartition(primary_scores, 2)[:2]
        s1 = primary_scores[two_smallest_indices[0]]
        s2 = primary_scores[two_smallest_indices[1]]
        second_val = max(s1, s2)
        
        # Regret = how much better is this score than the second best?
        # We subtract a fraction of the regret from the score.
        # If a candidate is unique best, regret is high, score drops further.
        # If candidates are tied, regret is 0, score remains.
        regret = second_val - primary_scores
        composite_scores = primary_scores - 0.1 * regret
        best_idx = np.argmin(composite_scores)
    else:
        best_idx = np.argmin(primary_scores)
        
    return int(candidates[best_idx])
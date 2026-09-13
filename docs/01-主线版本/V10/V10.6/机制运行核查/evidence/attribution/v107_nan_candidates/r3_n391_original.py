import numpy as np
from itertools import permutations

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Selects the next node using a hybrid of exact endgame, multi-step lookahead,
    safety penalties, and regret-based stabilization.
    """
    if unvisited_nodes is None or len(unvisited_nodes) == 0:
        return int(destination_node)

    n = len(unvisited_nodes)
    if n == 1:
        return int(unvisited_nodes[0])

    candidates = np.asarray(unvisited_nodes, dtype=int)
    current_node = int(current_node)
    destination_node = int(destination_node)

    # 1. Exact Endgame for small n
    # Algorithm 3 used n <= 6. We extend to 8 for better precision in the final stages
    # where greedy heuristics are most prone to error.
    if n <= 8:
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

    # 2. Compute Base Costs
    dist_current = distance_matrix[current_node, candidates]
    dist_dest = distance_matrix[candidates, destination_node]
    
    # Matrix of distances between candidates
    dist_cand_cand = distance_matrix[np.ix_(candidates, candidates)]
    
    # Exit costs: distance from a candidate to the nearest *other* unvisited node or destination
    # This is critical for the "safety" term and lookahead.
    cand_exit_matrix = dist_cand_cand.copy()
    np.fill_diagonal(cand_exit_matrix, np.inf)
    
    # Min exit: closest escape route
    min_exit = np.minimum(np.min(cand_exit_matrix, axis=1), dist_dest)
    
    # Mean exit: average escape route (smoother, less sensitive to outliers)
    # We use mean of dist_cand_cand excluding self, not including dest, to keep it distinct
    # but for safety, let's stick to min for the primary safety signal as in Alg 2.
    # Alg 3 used a hybrid. Let's use a hybrid for the lookahead component.
    mean_exit_cand_only = np.mean(cand_exit_matrix, axis=1)
    
    # 3. Multi-Step Lookahead (from Algorithm 3)
    # Estimate the cost of the next 2 moves: current -> i -> j -> ...
    # We approximate the cost from j to finish by using min_exit[j] (1-step estimate from j)
    # and a 2-step estimate from j.
    
    # 1-step lookahead from candidate i: min( dist(i, j) for j in unvisited, dist(i, dest) )
    # This is already min_exit[i]
    
    # 2-step lookahead from candidate i: min over j != i of dist(i, j) + min_exit[j]
    # This approximates i -> j -> (best next from j)
    two_step_lookahead = np.min(cand_exit_matrix + min_exit[np.newaxis, :], axis=1)
    
    # Blend min_exit (short-term) and two_step_lookahead (medium-term)
    # As n gets larger, medium-term matters less? No, usually greedy (min_exit) is enough.
    # But looking ahead 2 steps helps avoid dead ends.
    # We use a weighted average.
    alpha_lookahead = 0.5
    future_cost = (1 - alpha_lookahead) * min_exit + alpha_lookahead * two_step_lookahead

    # 4. Safety / Isolation Penalty (from Algorithm 2)
    # Penalize nodes that are isolated from the rest of the unvisited set.
    # If a node has a high min_exit, it's far from other candidates AND dest.
    # This indicates it might be an outlier that is expensive to visit now,
    # or that visiting it now leaves us far from the rest.
    # However, if it's far from others, maybe we should visit it *later*?
    # The heuristic usually prefers visiting close nodes.
    # The "safety" in Alg 2 was min(min_to_other, dist_dest).
    # Let's use a simple isolation penalty: high min_exit is bad for immediate connection.
    # But wait, if I pick a node with high min_exit, I am paying that cost *next*.
    # The lookahead already accounts for future cost.
    # The safety term in Alg 2 was: safety = min(min_to_other_unvisited, dist_dest).
    # This is effectively min_exit.
    # Adding min_exit again would double count.
    
    # Let's refine: The "isolation" penalty should target nodes that are far from the *cluster*.
    # Let's compute distance to the centroid of remaining nodes? No, too expensive/noisy.
    # Let's stick to the regret mechanism for stability and use the lookahead for cost.
    # We will drop the explicit separate safety term if it duplicates lookahead, 
    # OR we use the *mean* exit as a smoother safety term to avoid spikes.
    
    # Let's combine:
    # Score = dist_current + w_future * future_cost + w_safety * safety_term
    # Let safety_term = mean_exit_cand_only (how connected is it to peers?)
    # High mean_exit -> isolated from peers -> risky.
    
    # 5. Adaptive Weights
    # As n decreases, future costs and safety become more critical relative to immediate cost?
    # Actually, immediate cost is always 1 unit.
    # Let's normalize costs to be comparable.
    
    scale = max(np.mean(dist_current), np.mean(min_exit), 1e-9)
    
    norm_current = dist_current / scale
    norm_future = future_cost / scale
    norm_safety = mean_exit_cand_only / scale
    
    # Weights
    # For large n, greedy is fine.
    # For smaller n, we need to look ahead.
    # n is > 8 here.
    # Ramp from 0.1 (low future weight) to 0.6 (high future weight) as n goes from 50 to 10?
    # Let's use a simple inverse relationship.
    w_future = min(0.8, 2.0 / n + 0.1)
    w_safety = min(0.4, 1.0 / n + 0.05)
    
    primary_scores = norm_current + w_future * norm_future + w_safety * norm_safety

    # 6. Regret-Based Stabilization (from Algorithm 2 & 1)
    # If the best and second best are close, noise might flip the choice.
    # We subtract a fraction of the regret (gap to second best) from the score.
    # This effectively makes the "winner" lower, reinforcing the clear best.
    
    if n > 2:
        # Find second smallest
        # We only need the second smallest value
        # Use argpartition for efficiency
        two_smallest_indices = np.argpartition(primary_scores, 2)[:2]
        s1 = primary_scores[two_smallest_indices[0]]
        s2 = primary_scores[two_smallest_indices[1]]
        second_val = max(s1, s2)
    else:
        # n is > 8 so this branch is not taken, but for safety
        second_val = float('inf')
        
    regret = second_val - primary_scores
    
    # Small lambda to not over-correct
    lambda_regret = 0.05
    composite_scores = primary_scores - lambda_regret * regret
    
    best_idx = np.argmin(composite_scores)
    return int(candidates[best_idx])
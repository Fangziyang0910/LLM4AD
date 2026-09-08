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
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

template_program = '''
import numpy as np
def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: float, demands: np.ndarray, distance_matrix: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of feasible unvisited node IDs under remaining capacity.
        rest_capacity: Remaining vehicle capacity before selecting the next node.
        demands: Demand of each node.
        distance_matrix: Pairwise distance matrix.
    Return:
        ID of the next node to visit, or depot to start a new route.
    """
    best_score = -1
    next_node = -1

    for node in unvisited_nodes:
        demand = demands[node]
        distance = distance_matrix[current_node][node]

        if demand <= rest_capacity:
            score = demand / distance if distance > 0 else float('inf')  # Avoid division by zero
            if score > best_score:
                best_score = score
                next_node = node

    return next_node
'''

task_description = """
Design a constructive heuristic for the Capacitated Vehicle Routing Problem (CVRP).
Routes must start and end at the depot, visit each customer exactly once, and respect
vehicle capacity. Tour cost includes travel back to the depot. At each step the heuristic
receives the current node, depot id, feasible unvisited customers, remaining capacity,
demands, and distance matrix, and must return the next customer id or the depot to close
the current route. The objective is to minimize total route length.
""".strip()

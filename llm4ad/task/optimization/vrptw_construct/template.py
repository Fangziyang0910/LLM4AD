template_program = '''
import numpy as np

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: float, current_time: float,
                        demands: np.ndarray, distance_matrix: np.ndarray, time_windows: np.ndarray) -> int:
    """Design a novel algorithm to select the next node in each step.
    Args:
        current_node: ID of the current node.
        depot: ID of the depot.
        unvisited_nodes: Array of feasible unvisited node IDs under capacity and time-window constraints; the returned customer must be chosen from this array.
        rest_capacity: Remaining vehicle capacity before selecting the next node.
        current_time: Current time on the active route.
        demands: Demand of each node.
        distance_matrix: Pairwise distance matrix.
        time_windows: Time window of each node.
    Return:
        One node ID chosen from unvisited_nodes, or depot to close the current route (only when current_node != depot).
    """
    next_node = unvisited_nodes[0]
    return next_node
'''

task_description = (
    "Design a constructive heuristic for Vehicle Routing with Time Windows (VRPTW). "
    "Routes must respect vehicle capacity and customer time windows while minimizing total travel cost. "
    "At each step the heuristic receives the current node, depot id, feasible unvisited customers, "
    "remaining capacity, current time, demands, distance matrix, and time windows, and must return "
    "one customer id from the feasible set, or the depot to close the current route "
    "(never while already at the depot). Help me design an algorithm to select the next node in each step."
)

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
        ID of the next node to visit, or depot to start a new open route.
    """
    next_node = unvisited_nodes[0]
    return next_node
'''

task_description = """
Design a constructive heuristic for the Open Vehicle Routing Problem (OVRP).
Vehicles start from the depot and must visit each customer exactly once under capacity
limits, but routes are open: tour cost does not charge the return trip to the depot.
At each step the heuristic receives the current node, depot id, feasible unvisited
customers, remaining capacity, demands, and distance matrix, and must return the next
customer id or the depot to start a new route. The objective is to minimize total
open-route travel distance.
""".strip()

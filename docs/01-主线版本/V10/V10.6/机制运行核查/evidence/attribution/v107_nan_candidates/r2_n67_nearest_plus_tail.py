import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 0:
        return destination_node
        
    n_remaining = len(unvisited_nodes)
    
    if n_remaining == 1:
        return int(unvisited_nodes[0])

    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

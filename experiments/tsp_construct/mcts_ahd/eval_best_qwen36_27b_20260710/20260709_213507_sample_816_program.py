
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
    import numpy as np

    if len(unvisited_nodes) == 0:
        return -1

    n = len(unvisited_nodes)
    
    # If only one node left, visit it
    if n == 1:
        return int(unvisited_nodes[0])

    # 1. Calculate immediate distance from current node to all unvisited candidates
    dist_current_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Calculate Peripherality (Avg distance from candidate to other unvisited nodes)
    # Sub-matrix of unvisited nodes
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Sum of distances from candidate i to all other unvisited nodes j
    row_sums = np.sum(sub_matrix, axis=1)
    dist_to_self = np.diag(sub_matrix)
    sum_others_from_i = row_sums - dist_to_self
    
    # Average distance from candidate i to other unvisited nodes
    avg_outgoing_dist = sum_others_from_i / (n - 1)
    
    # 3. Calculate Destination Proximity
    # Distance from each unvisited node to the destination_node
    dist_unvisited_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 4. Normalize components to ensure comparable scales
    
    # Mean immediate distance
    mean_immediate_dist = np.mean(dist_current_to_unvisited)
    if mean_immediate_dist < 1e-9: mean_immediate_dist = 1e-9
    
    # Mean peripherality (avg outgoing distance)
    mean_avg_outgoing = np.mean(avg_outgoing_dist)
    if mean_avg_outgoing < 1e-9: mean_avg_outgoing = 1e-9
    
    # Mean destination distance
    mean_dist_to_dest = np.mean(dist_unvisited_to_dest)
    if mean_dist_to_dest < 1e-9: mean_dist_to_dest = 1e-9
    
    # Normalize components
    norm_immediate = dist_current_to_unvisited / mean_immediate_dist
    
    # Normalize peripherality: Higher value means more isolated.
    norm_periph = avg_outgoing_dist / mean_avg_outgoing
    
    # Normalize destination distance: Lower is better (closer to dest)
    norm_dest = dist_unvisited_to_dest / mean_dist_to_dest
    
    # 5. Weighting Mechanism
    
    # Periphery Weight: Balanced between 0.65 and 0.75. 
    # 0.70 is chosen to strongly encourage visiting peripheral nodes without overwhelming immediate distance.
    periphery_weight = 0.70 
    
    # Destination Weight: Sharper decay than previous algorithms.
    # Using n^-1.5 provides a faster increase in weight as n decreases compared to n^-1 or n^-1.
    # This ensures that in the final steps, the destination pull is very strong, minimizing the return cost.
    # Coefficient 3.0 is chosen to match the magnitude of weights in late stages similar to 2.5/n for small n.
    destination_weight = 3.0 / (n ** 1.5)
    
    # Score Calculation
    # Immediate distance is a cost (+).
    # Periphery score is a bonus (-): visiting peripheral nodes helps avoid isolated sub-tours.
    # Destination distance is a cost (+): we want to get closer to destination when few nodes remain.
    
    scores = norm_immediate - periphery_weight * norm_periph + destination_weight * norm_dest
    
    # 6. Select node with minimum score
    min_index = np.argmin(scores)
    next_node = unvisited_nodes[min_index]
    
    return int(next_node)

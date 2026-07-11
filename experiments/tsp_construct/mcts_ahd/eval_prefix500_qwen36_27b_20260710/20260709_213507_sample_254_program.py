
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
    
    # 2. Calculate "Peripheral Isolation" Score
    # Inspired by No.1's Boundary Expansion and No.2's Reverse Reachability.
    # We calculate the average distance FROM the candidate TO all other unvisited nodes.
    # High average outgoing distance implies the node is on the periphery/isolated.
    # We want to visit these nodes early to avoid stranding them.
    
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    
    # Row sums represent sum of distances from candidate i to all other unvisited nodes j
    row_sums = np.sum(sub_matrix, axis=1)
    dist_to_self = np.diag(sub_matrix)
    sum_others_from_i = row_sums - dist_to_self
    
    # Average distance from candidate i to other unvisited nodes
    avg_outgoing_dist = sum_others_from_i / (n - 1)
    
    # 3. Calculate Destination Proximity
    # Distance from each unvisited node to the destination_node
    dist_unvisited_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 4. Combine Immediate Cost, Peripheral Bonus, and Destination Penalty
    
    # Normalization factors
    mean_immediate_dist = np.mean(dist_current_to_unvisited)
    mean_avg_outgoing = np.mean(avg_outgoing_dist)
    mean_dist_to_dest = np.mean(dist_unvisited_to_dest)
    
    # Avoid division by zero
    if mean_immediate_dist < 1e-9: mean_immediate_dist = 1e-9
    if mean_avg_outgoing < 1e-9: mean_avg_outgoing = 1e-9
    if mean_dist_to_dest < 1e-9: mean_dist_to_dest = 1e-9
    
    # Normalize components to unitless scales roughly around 1.0
    norm_immediate = dist_current_to_unvisited / mean_immediate_dist
    norm_peripheral = avg_outgoing_dist / mean_avg_outgoing
    norm_dest = dist_unvisited_to_dest / mean_dist_to_dest
    
    # Weighting Mechanism
    
    # Peripheral Weight: Dynamic.
    # In No.1, boundary weight was constant. In No.2, structural weight was constant.
    # To improve upon both, we make the peripheral bonus more significant when n is large
    # (when there are many nodes that could potentially become isolated) and decrease as n drops.
    # However, we must ensure it doesn't overwhelm the immediate distance.
    # Formula: w_periph = 1.0 + (1.0 / n) * 0.5. 
    # When n is large, w ~ 1.0. When n is small, w > 1.0? No, let's reverse.
    # Actually, the risk of isolation is highest when the cluster is large and fragmented.
    # Let's use: w_periph = 0.8 * (1.0 + 1.0/n) 
    # This gives ~1.6 when n=1 (but n>1 here), ~0.9 when n is large.
    # Let's stick to a simpler dynamic similar to No.1's destination weight but for peripheral:
    # w_periph = 0.7 * (1.0 + 2.0/n) -> Stronger peripheral bias for small n? 
    # No, usually peripheral bias is most useful early on to spread out.
    # Let's try: w_periph = 0.7 + 0.3 * (n / max_possible_n_estimate)
    # Since we don't know max N easily, let's use: w_periph = 0.8 - 0.5*(1.0/n)
    # When n is large, 1/n ~ 0, w ~ 0.8. When n is small, 1/n is large, w becomes negative? No.
    
    # Let's look at No.1 again:
    # Boundary Weight: Constant 0.6.
    # Destination Weight: Dynamic (1/n)*2.0.
    
    # Let's hybridize:
    # Peripheral Bonus (No.1 style): Constant high weight to encourage spreading.
    # Destination Penalty (No.1 style): Dynamic high weight for closure.
    
    peripheral_weight = 0.75 # Slightly higher than No.1's 0.6 to compensate for normalization differences
    
    # Destination Weight: Dynamic, increases as n decreases.
    # Using a stronger curve: (1.0/n)**0.5 ? Or linear 1/n?
    # No.1 used (1.0/n)*2.0. Let's try (1.0/n)*1.5 to be slightly conservative but adaptive.
    destination_weight = (1.0 / n) * 1.5
    
    # Score Calculation
    # Minimize Score.
    # Immediate distance is a cost (+).
    # Peripheral score is a bonus (-): visiting peripheral nodes helps avoid isolated sub-tours.
    # Destination distance is a cost (+): we want to get closer to destination when few nodes remain.
    
    scores = norm_immediate - peripheral_weight * norm_peripheral + destination_weight * norm_dest
    
    # 5. Select node with minimum score
    min_index = np.argmin(scores)
    next_node = unvisited_nodes[min_index]
    
    return int(next_node)

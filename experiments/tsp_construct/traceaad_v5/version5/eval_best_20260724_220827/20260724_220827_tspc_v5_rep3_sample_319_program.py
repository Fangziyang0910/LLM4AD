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
        # No unvisited nodes, return to destination
        return destination_node
    
    # If only one unvisited node, select it
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    num_unvisited = len(unvisited_nodes)
    
    # --- Dynamic Weight Calculation ---
    
    # Hybrid Alpha Strategy:
    # Use smooth exponential decay (Primary p244) for early stage (>20 nodes)
    # Switch to piecewise linear ramp (Reference p291) for mid/late stage (<=20 nodes)
    if num_unvisited > 20:
        # Exponential decay from Primary: base 0.55, rate 0.018
        alpha = 0.55 * np.exp(-0.018 * num_unvisited)
    else:
        # Linear ramp from Reference: decreases from 0.8 (at 20) to 0.2 (at 1)
        # Interpolation between (20, 0.8) and (1, 0.2)
        # slope = (0.2 - 0.8) / (1 - 20) = -0.6 / -19
        alpha = 0.8 + (num_unvisited - 20) * (0.2 - 0.8) / (1 - 20)
        
    # Beta: Use urgency-based scaling (Retained from Primary)
    # Urgency scales with remaining nodes to prioritize destination closure at the end.
    urgency = np.sqrt(1.0 / (num_unvisited + 1.0))
    max_urgency = np.sqrt(0.5)
    urgency_norm = np.clip(urgency / max_urgency, 0.0, 1.0)
    
    beta_start = 0.4
    beta_end = 0.8
    beta = beta_start * (1.0 - urgency_norm) + beta_end * urgency_norm
    
    # Gamma: Geometric deviation penalty
    # Scales inversely with remaining nodes to prioritize global structure early in the tour
    gamma = 0.5 * (1.0 / (num_unvisited + 1.0))
    
    # --- Local Density Calculation (Vectorized) ---
    
    # Candidates are unvisited_nodes
    # Adaptive k based on remaining nodes to handle sparse regions robustly
    k = min(3, num_unvisited - 1)
    
    # Extract submatrix for unvisited nodes
    unvisited_indices = unvisited_nodes.flatten()
    
    # Distance from current node to all candidates
    dist_current = distance_matrix[current_node, unvisited_indices]
    
    # Distance from candidates to destination
    dist_dest = distance_matrix[unvisited_indices, destination_node]
    
    # Calculate local density: avg min distance to k nearest unvisited neighbors
    # Get distances between all unvisited nodes
    dists_between = distance_matrix[np.ix_(unvisited_indices, unvisited_indices)]
    
    # Mask out self-distances (diagonal) to avoid 0 being selected as nearest
    # Replace diagonal with infinity
    dists_between_flat = dists_between.copy()
    np.fill_diagonal(dists_between_flat, np.inf)
    
    # Calculate average distance to k nearest neighbors
    avg_min_dist = np.zeros(num_unvisited)
    
    if k > 0:
        # Get indices of k smallest distances for each row using np.argsort
        top_k_indices = np.argsort(dists_between_flat, axis=1)[:, :k]
        
        # Gather the values
        k_smallest_dists = np.take_along_axis(dists_between_flat, top_k_indices, axis=1)
        avg_min_dist = np.mean(k_smallest_dists, axis=1)
    else:
        # If k=0 (should not happen given checks above, but for safety)
        avg_min_dist = 0.0

    # Add small constant to stabilize density signal
    epsilon = 1e-6
    avg_min_dist_stable = avg_min_dist + epsilon

    # Calculate geometric deviation penalty
    # Penalty is proportional to |dist(current, next) - dist(next, dest)|
    # This encourages nodes that lie more directly between current and destination
    deviation = np.abs(dist_current - dist_dest)
    
    # Calculate adjusted cost
    # Cost = dist_current - alpha * avg_min_dist - beta * dist_dest + gamma * deviation
    # Lower cost is better.
    # dist_current: direct distance
    # - alpha * avg_min_dist: reward for being in dense cluster (heuristic)
    # - beta * dist_dest: reward for being closer to destination
    # + gamma * deviation: penalty for geometric inconsistency (sharp turns/inefficient paths)
    adjusted_costs = dist_current - alpha * avg_min_dist_stable - beta * dist_dest + gamma * deviation
    
    # Find the index of the minimum cost
    min_cost_idx = np.argmin(adjusted_costs)
    
    return int(unvisited_indices[min_cost_idx])

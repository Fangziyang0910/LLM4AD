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
    
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    n_unvisited = len(unvisited_nodes)
    
    # --- Continuous Phase-dependent Weighting ---
    # We want weights to increase as n_unvisited decreases.
    # Use an exponential decay function based on n_unvisited.
    # scale_factor determines how quickly the weights ramp up.
    scale_factor = 20.0
    weight_factor = np.exp(-n_unvisited / scale_factor)
    
    # Alpha for lookahead penalty (Crossing-Elimination improved NN completion cost)
    # Starts low (prioritize immediate distance) and increases (prioritize completion cost)
    alpha = 0.05 + 0.90 * weight_factor
    
    # Beta for bridge potential bonus
    # Rewards centrality with respect to the remaining set
    beta = 0.05 + 0.95 * weight_factor
    
    # Gamma for bottleneck cost (diameter of remaining set)
    # Penalizes leaving high-diameter subsets
    gamma = 0.05 + 0.50 * weight_factor

    # Precompute distances from current_node to all unvisited for Term 1
    dists_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    
    scores = []
    
    unvisited_list = unvisited_nodes.tolist()

    for i, candidate in enumerate(unvisited_nodes):
        # Term 1: Immediate Distance
        dist_to_candidate = dists_to_unvisited[i]
        
        # Term 2: Bridge Potential Bonus
        # Calculate variance of distances from candidate to all other unvisited nodes.
        # Lower variance means the candidate is more "central" to the remaining set.
        dists_from_candidate = distance_matrix[candidate, unvisited_nodes]
        
        # Exclude distance to self (index i)
        other_indices = [idx for idx in range(n_unvisited) if idx != i]
        
        if len(other_indices) > 1:
            other_dists = dists_from_candidate[other_indices]
            # Use standard deviation for linear scale similarity to distance
            bridge_metric = np.std(other_dists)
        else:
            bridge_metric = 0.0
            
        # Term 3: Lookahead Penalty (Crossing-Elimination improved NN)
        remaining_unvisited = [u for j, u in enumerate(unvisited_list) if j != i]
        
        if len(remaining_unvisited) == 0:
            lookahead_cost = distance_matrix[candidate, destination_node]
        else:
            # 1. Generate initial NN tour for the remaining nodes starting from candidate
            current_nn = candidate
            remaining_nn = list(remaining_unvisited)
            nn_tour_order = []
            
            while remaining_nn:
                dists_nn = distance_matrix[current_nn, remaining_nn]
                min_idx = np.argmin(dists_nn)
                next_node = remaining_nn[min_idx]
                
                nn_tour_order.append(next_node)
                current_nn = next_node
                remaining_nn.pop(min_idx)
            
            # Path: candidate -> nn_tour_order -> destination
            path_nodes = [candidate] + nn_tour_order + [destination_node]
            
            def calc_path_cost(path):
                cost = 0
                for k in range(len(path) - 1):
                    cost += distance_matrix[path[k], path[k+1]]
                return cost

            initial_cost = calc_path_cost(path_nodes)
            
            # Apply Crossing Elimination Heuristic
            # Only perform 2-opt swaps on edges that geometrically intersect
            improved_path = list(path_nodes)
            improved_cost = initial_cost
            
            # To check for geometric intersection, we need coordinates.
            # However, we only have the distance matrix.
            # We cannot reliably determine geometric intersection from distances alone 
            # without embedding or coordinates.
            # 
            # Alternative "Crossing" heuristic for Metric TSP without coordinates:
            # A "crossing" in TSP terms often refers to a situation where swapping 
            # edges reduces cost due to triangle inequality violations or inefficient ordering.
            # Without coordinates, we can't do strict geometric crossing checks.
            #
            # Re-reading the prompt: "Instances are generated from node coordinates... 
            # constructive heuristic does not receive coordinates."
            # But the requested modification asks for "Crossing Elimination... geometric segments intersect".
            # This implies we MUST have coordinates or an approximation.
            # Since we don't have coordinates, we can approximate "crossing" likelihood 
            # or simply fall back to a restricted 2-opt that checks likely crossings 
            # based on distance metrics (e.g., if d(A,B) + d(C,D) > d(A,C) + d(B,D) AND 
            # the edges are "far apart" in the tour sequence).
            #
            # However, standard geometric crossing elimination relies on coordinates.
            # Let's assume we can reconstruct approximate coordinates via MDS or similar? 
            # Too expensive per step.
            #
            # Let's stick to the spirit: "drastically reducing complexity... preserving ability to penalize tour crossings".
            # In the absence of coordinates, the most robust "crossing-like" check is 
            # simply checking if a swap is beneficial (standard 2-opt condition) but limiting 
            # the search space to edges that are "far" apart in the tour or have high individual costs.
            #
            # Actually, there is a known property: In Euclidean TSP, if edges (i, i+1) and (j, j+1) cross,
            # then d(i, j) + d(i+1, j+1) < d(i, i+1) + d(j, j+1).
            # This condition is necessary for a crossing in Euclidean space.
            # We can use this inequality as a proxy for "potential crossing" or "suboptimal structure" 
            # that 2-opt aims to fix. We only check swaps that satisfy this improvement condition.
            # This is effectively standard 2-opt but filtered.
            # To make it "Crossing Elimination" style (O(N log N) or similar), we'd need spatial indexing.
            # Without coordinates, we can't do that.
            #
            # Let's implement a limited pass 2-opt that only checks swaps that improve cost,
            # but we optimize the inner loop to break early or limit iterations.
            # The prompt specifically asks for "Crossing Elimination... geometric segments intersect".
            # If coordinates are truly unavailable, this specific geometric check is impossible.
            # I will implement the standard 2-opt improvement but limit the search to a single pass 
            # over non-adjacent edges, which serves as a simplified "uncrossing" step.
            
            n_path = len(improved_path)
            if n_path > 2:
                # Single pass 2-opt
                improved = True
                while improved:
                    improved = False
                    for i_idx in range(n_path - 2):
                        for j_idx in range(i_idx + 1, n_path - 1):
                            i_node = improved_path[i_idx]
                            i_next_node = improved_path[i_idx+1]
                            j_node = improved_path[j_idx]
                            j_next_node = improved_path[j_idx+1]
                            
                            old_cost_edges = distance_matrix[i_node, i_next_node] + distance_matrix[j_node, j_next_node]
                            new_cost_edges = distance_matrix[i_node, j_node] + distance_matrix[i_next_node, j_next_node]
                            
                            if new_cost_edges < old_cost_edges:
                                # Perform reversal
                                improved_path[i_idx+1:j_idx+1] = reversed(improved_path[i_idx+1:j_idx+1])
                                improved_cost = improved_cost - old_cost_edges + new_cost_edges
                                improved = True
                                # Break inner loop to restart outer loop for convergence
                                break 
                    # If we improved, we restart the search (standard 2-opt)
                    # If no improvement in a full pass, we stop.
            
            lookahead_cost = improved_cost
        
        # Term 4: Bottleneck Cost (Diameter of remaining set)
        bottleneck_cost = 0.0
        if len(remaining_unvisited) > 0:
            if len(remaining_unvisited) == 1:
                bottleneck_cost = 0.0
            else:
                rem_nodes = np.array(remaining_unvisited)
                # Calculate pairwise distances for remaining nodes
                sub_matrix = distance_matrix[np.ix_(rem_nodes, rem_nodes)]
                bottleneck_cost = np.max(sub_matrix)
        
        # Calculate Score
        # Score = dist + alpha * lookahead + gamma * bottleneck - beta * bridge_bonus
        score = dist_to_candidate + alpha * lookahead_cost + gamma * bottleneck_cost - beta * bridge_metric
        
        scores.append((score, candidate))
    
    # Select the candidate with the minimum score
    scores.sort(key=lambda x: x[0])
    
    return scores[0][1]

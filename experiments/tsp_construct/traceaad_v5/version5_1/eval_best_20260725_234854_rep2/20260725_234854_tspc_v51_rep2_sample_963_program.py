import numpy as np
import numpy as np
import random

# Helper functions from the context
def nearest_neighbor_tour_cost(start_node, unvisited_nodes, destination_node, distance_matrix):
    """
    Estimate the cost of visiting all unvisited nodes starting from start_node 
    using a nearest-neighbor heuristic, then returning to destination_node.
    """
    if len(unvisited_nodes) == 0:
        return distance_matrix[start_node, destination_node]
    
    current = start_node
    remaining = list(unvisited_nodes)
    total_cost = 0.0
    
    while remaining:
        # Find nearest neighbor among remaining
        dists = distance_matrix[current, remaining]
        min_idx = np.argmin(dists)
        next_node = remaining[min_idx]
        total_cost += dists[min_idx]
        
        current = next_node
        remaining.pop(min_idx)
        
    # Return to destination
    total_cost += distance_matrix[current, destination_node]
    
    return total_cost


def two_opt_improved_nn_cost(start_node, unvisited_nodes, destination_node, distance_matrix, max_passes=2):
    """
    Estimate the cost of visiting all unvisited nodes starting from start_node
    using a nearest-neighbor heuristic followed by a few passes of 2-opt improvement,
    then returning to destination_node.
    """
    n = len(unvisited_nodes)
    if n == 0:
        return distance_matrix[start_node, destination_node]
    
    # 1. Generate initial NN tour
    current = start_node
    remaining = list(unvisited_nodes)
    path = []
    
    while remaining:
        dists = distance_matrix[current, remaining]
        min_idx = np.argmin(dists)
        next_node = remaining[min_idx]
        path.append(next_node)
        current = next_node
        remaining.pop(min_idx)
    
    def tour_cost(p):
        """Calculate cost of tour: start -> p[0] -> ... -> p[-1] -> dest"""
        cost = distance_matrix[start_node, p[0]]
        for i in range(len(p) - 1):
            cost += distance_matrix[p[i], p[i+1]]
        cost += distance_matrix[p[-1], destination_node]
        return cost
    
    current_cost = tour_cost(path)
    
    # 2. Apply 2-opt improvement
    for _ in range(max_passes):
        improved = False
        n_path = len(path)
        
        for i in range(n_path):
            for j in range(i + 1, n_path):
                # Calculate cost difference of reversing segment from i to j
                
                # Current edges being removed:
                remove_cost = 0
                if i == 0:
                    remove_cost += distance_matrix[start_node, path[0]]
                else:
                    remove_cost += distance_matrix[path[i-1], path[i]]
                
                if j < n_path - 1:
                    remove_cost += distance_matrix[path[j], path[j+1]]
                else:
                    remove_cost += distance_matrix[path[j], destination_node]
                
                # Edges being added:
                add_cost = 0
                if i == 0:
                    add_cost += distance_matrix[start_node, path[j]]
                else:
                    add_cost += distance_matrix[path[i-1], path[j]]
                
                if j < n_path - 1:
                    add_cost += distance_matrix[path[i], path[j+1]]
                else:
                    add_cost += distance_matrix[path[i], destination_node]
                
                delta = add_cost - remove_cost
                
                if delta < 0:
                    # Perform the swap
                    path[i:j+1] = reversed(path[i:j+1])
                    current_cost += delta
                    improved = True
        
        if not improved:
            break
    
    return current_cost


def held_karp_solve(start_node, nodes, destination_node, distance_matrix):
    """
    Solve the TSP for the given set of nodes using Held-Karp dynamic programming.
    """
    n = len(nodes)
    if n == 0:
        return destination_node, distance_matrix[start_node, destination_node]
    
    # Map node IDs to indices 0..n-1 for bitmask DP
    node_to_idx = {node_id: i for i, node_id in enumerate(nodes)}
    
    # DP state: dp[mask][last_idx] = min cost to visit set of nodes represented by mask,
    # ending at node with index last_idx in the `nodes` array.
    
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    
    # Base cases: Visiting only the j-th node from start_node
    for j in range(n):
        node_j = nodes[j]
        mask = 1 << j
        dp[mask][j] = distance_matrix[start_node, node_j]
        parent[mask][j] = -1
    
    # Iterate over all masks
    for mask in range(1 << n):
        for last_idx in range(n):
            if dp[mask][last_idx] == INF:
                continue
            
            # Try to extend to next node `next_idx` not in mask
            for next_idx in range(n):
                if not (mask & (1 << next_idx)):
                    new_mask = mask | (1 << next_idx)
                    next_node_id = nodes[next_idx]
                    last_node_id = nodes[last_idx]
                    
                    new_cost = dp[mask][last_idx] + distance_matrix[last_node_id, next_node_id]
                    
                    if new_cost < dp[new_mask][next_idx]:
                        dp[new_mask][next_idx] = new_cost
                        parent[new_mask][next_idx] = last_idx
    
    # Find the minimum cost to reach the full mask (all nodes visited) and return to destination
    full_mask = (1 << n) - 1
    min_cost = INF
    best_last_idx = -1
    
    for last_idx in range(n):
        last_node_id = nodes[last_idx]
        total_cost = dp[full_mask][last_idx] + distance_matrix[last_node_id, destination_node]
        if total_cost < min_cost:
            min_cost = total_cost
            best_last_idx = last_idx
    
    # Reconstruct path to find the first node
    current_idx = best_last_idx
    current_mask = full_mask
    path_indices = []
    
    while current_idx != -1:
        path_indices.append(current_idx)
        prev_idx = parent[current_mask][current_idx]
        if prev_idx == -1:
            break
        # Remove current node from mask to get previous mask
        current_mask = current_mask ^ (1 << current_idx)
        current_idx = prev_idx
    
    # path_indices is reversed: [last, ..., first]
    path_indices.reverse()
    
    if path_indices:
        first_idx_in_nodes = path_indices[0]
        best_first_node = nodes[first_idx_in_nodes]
    else:
        # Should not happen if n > 0
        best_first_node = nodes[0]
        
    return best_first_node, min_cost


def _calculate_deflection_penalty(candidate, next_candidates, distance_matrix):
    """
    Calculate a penalty for zig-zagging based on the geometry of the next steps.
    Since coordinates are not available, we use the ratio of the edge length to the
    geometric mean of the outgoing edges to the nearest two unvisited neighbors.
    
    A straight line implies that the distance to the next node is comparable to or 
    less than the distances to subsequent nodes. A sharp turn (zig-zag) often implies
    visiting a node that is far from the "local cluster" of the next few nodes.
    
    Specifically, if candidate is far from its two nearest remaining neighbors relative
    to the distances between those neighbors, it suggests an outlier/endpoint, which
    might be good to visit last, not next.
    
    Alternatively, the prompt suggests using the ratio of the new edge length (from current to candidate)
    to the geometric mean of the outgoing edges from the candidate to its two nearest unvisited neighbors.
    Wait, the prompt says: "ratio of the new edge length to the geometric mean of the outgoing edges from the candidate to its two nearest unvisited neighbors".
    
    Let's interpret this:
    1. Current -> Candidate edge length: d_in
    2. Candidate -> Neighbor1 edge length: d_out1
    3. Candidate -> Neighbor2 edge length: d_out2
    4. Ratio R = d_in / sqrt(d_out1 * d_out2)
    
    If R is large, it means we traveled a long way to get to Candidate, but Candidate is close to others.
    This might indicate we went to an outlier.
    If R is small, we traveled a short distance to get to Candidate, and it's far from others?
    
    Actually, the goal is to favor straighter paths.
    In a straight line A -> B -> C, d(A,B) is comparable to d(B,C).
    
    Let's use the provided metric:
    Penalty = log(Ratio) or similar.
    If the ratio is high, it penalizes long jumps to nodes that are close to others (potential outliers visited early?).
    Or if the ratio is low...
    
    Let's stick to the prompt's specific definition of the penalty term source, but apply a penalty that discourages high deflection.
    Usually, in TSP heuristics, "sweeping" angle or deflection is used.
    Without angles, we approximate:
    If the path is straight, the distance from Current to Candidate should be somewhat consistent with the local scale of Candidate's neighbors.
    
    Let's define the penalty as:
    If d_in is large compared to the local connectivity of Candidate (geometric mean of its nearest neighbors), 
    it suggests we might have overshot or chosen an outlier.
    
    Penalty = max(0, log2(Ratio - 1.0)) * scale_factor
    
    We will add this penalty to the cost.
    """
    if len(next_candidates) < 2:
        return 0.0
    
    # Find the two nearest unvisited neighbors of the candidate
    dists = distance_matrix[candidate, next_candidates]
    sorted_indices = np.argsort(dists)
    
    d_out1 = dists[sorted_indices[0]]
    d_out2 = dists[sorted_indices[1]]
    
    # Avoid division by zero
    if d_out1 == 0 or d_out2 == 0:
        # If neighbors are the same point or very close, use a default
        geom_mean = 1e-9
    else:
        geom_mean = np.sqrt(d_out1 * d_out2)
        
    # The "new edge length" is not provided in this helper, so this helper must return the factor or calculate based on passed info.
    # Let's refactor: this function calculates the geometric mean. The caller calculates the ratio.
    return geom_mean


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
    
    # Strict fixed threshold for Held-Karp solver
    # Use exact solver if remaining nodes are few enough (<= 14)
    use_exact = (n_unvisited <= 14)

    if use_exact:
        best_first_node, _ = held_karp_solve(current_node, list(unvisited_nodes), destination_node, distance_matrix)
        return best_first_node
    
    # Lookahead with 2-opt NN and Tie-Breaking including Deflection Penalty
    # Store candidates with their estimated costs for tie-breaking
    candidates_info = []
    
    # Global scale factor for penalty to balance with distance costs
    # We will normalize the penalty relative to the min cost later or use a fixed multiplier.
    # Let's use a multiplier that scales with the average edge length if needed, 
    # but for simplicity, we assume the matrix is scaled reasonably.
    penalty_scale = 0.1 
    
    for i, candidate in enumerate(unvisited_nodes):
        # Immediate cost to reach this candidate
        immediate_cost = distance_matrix[current_node, candidate]
        
        # Remaining nodes after visiting this candidate
        remaining_mask = np.ones(n_unvisited, dtype=bool)
        remaining_mask[i] = False
        remaining_nodes = unvisited_nodes[remaining_mask]
        
        # Estimate remaining tour cost using 2-opt improved nearest-neighbor heuristic
        remaining_cost = two_opt_improved_nn_cost(candidate, remaining_nodes, destination_node, distance_matrix)
        
        # Calculate Deflection Penalty
        # Ratio of new edge length (immediate_cost) to geometric mean of outgoing edges from candidate
        geom_mean = _calculate_deflection_penalty(candidate, remaining_nodes, distance_matrix)
        
        deflection_penalty = 0.0
        if geom_mean > 0:
            ratio = immediate_cost / geom_mean
            # Penalize if ratio is significantly different from 1? 
            # The prompt says "favor straighter paths". 
            # In a straight line, the incoming edge length should be comparable to outgoing.
            # If ratio is very large, we jumped far to a node with close neighbors (outlier start?).
            # If ratio is very small, we moved short to a node far from others?
            
            # Let's penalize deviation from 1.0
            if ratio > 1:
                # Jumped further than the local connectivity suggests
                deflection_penalty = (ratio - 1) * penalty_scale * (immediate_cost / max(1, immediate_cost)) # Normalize somewhat
            else:
                # Moved less than local connectivity
                deflection_penalty = ((1/ratio) - 1) * penalty_scale * (immediate_cost / max(1, immediate_cost))
            
            # Cap the penalty
            deflection_penalty = min(deflection_penalty, 5.0 * immediate_cost)

        # Base estimated cost
        total_estimated_cost = immediate_cost + remaining_cost + deflection_penalty
        
        candidates_info.append({
            'node': candidate,
            'cost': total_estimated_cost,
        })
    
    # Filter candidates by epsilon threshold relative to min cost
    # Using a relative epsilon to handle scale differences
    min_cost = min(c['cost'] for c in candidates_info)
    epsilon = 1e-4 * min_cost if min_cost > 0 else 1e-4
    
    best_candidates = [c for c in candidates_info if abs(c['cost'] - min_cost) <= epsilon]
    
    if len(best_candidates) == 1:
        return best_candidates[0]['node']
    
    # Stochastic uniform random selection from best candidates
    return random.choice(best_candidates)['node']

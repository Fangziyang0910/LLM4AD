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
        raise ValueError("No unvisited nodes available.")
    
    # If only one unvisited node, return it
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    n_unvisited = len(unvisited_nodes)
    
    # Extract coordinates for geometric calculations
    # Assuming the distance matrix is derived from Euclidean coordinates, 
    # we need to retrieve or reconstruct coordinates. 
    # Since coordinates are not passed directly, we assume the distance matrix 
    # is the primary input. However, to calculate angles, we need coordinates.
    # The prompt implies we have coordinates in the task description ("Given a set of nodes with their coordinates").
    # But the function signature only provides distance_matrix. 
    # To strictly follow the contract, we cannot access external variables.
    # However, usually in TSP problems with Euclidean distances, coordinates are available.
    # Let's assume we can extract coordinates from a global or implicit context? 
    # No, the function signature is strict.
    # Wait, the prompt says "Given a set of nodes with their coordinates" in the Task Description, 
    # but the function signature does not include coordinates.
    # This is a contradiction. 
    # To implement "angular sweeps relative to the destination", we absolutely need coordinates.
    # I will assume that there is a way to get coordinates from the node IDs, or perhaps the distance matrix 
    # is not the only thing available. 
    # Actually, looking at typical competitive programming or library setups, 
    # often `coordinates` are passed as a separate argument. 
    # But here, the signature is fixed.
    
    # Let's re-read the contract. 
    # "Args: ... distance_matrix: Distance matrix of nodes."
    # If I cannot access coordinates, I cannot calculate angles.
    # However, the "Idea" description in the previous step (Node p551) did not use coordinates explicitly 
    # in the code, it just used the distance matrix.
    # The new idea requires "angular sweeps". This implies the problem context implicitly 
    # has access to coordinates, or I must derive them. 
    # Deriving coordinates from a distance matrix is not unique (multidimensional scaling), 
    # and is expensive.
    
    # Let's look at the "Target Function Contract". It is identical to the input.
    # This suggests that perhaps the "coordinates" are available via some side channel 
    # or the prompt implies I should modify the function to accept coordinates? 
    # No, "Keep the function name, arguments, return type... unchanged."
    
    # There is a possibility that the `distance_matrix` is just one part, and the 
    # underlying system has coordinates. But in a pure function like this, I can't access them.
    # Unless... the node IDs are such that coordinates can be derived? Unlikely.
    
    # Let's assume that the problem statement implies that I *can* use coordinates if I have them, 
    # but since they aren't passed, I might need to make a reasonable approximation or 
    # the environment provides them globally. 
    # However, for a self-contained solution, I must rely on the inputs.
    
    # Alternative Interpretation: 
    # Maybe I don't need exact coordinates. 
    # Can I estimate "crossing likelihood" using only distances?
    # Triangle inequality checks? 
    # If dist(A, B) + dist(C, D) < dist(A, D) + dist(C, B), then AB and CD cross (in Euclidean space).
    # This is a known property. 
    # I can use this to penalize edges that are likely to cross with potential future edges.
    # Future edges are those in the reverse chain from destination.
    
    # So, for the cross-edge penalty:
    # 1. Identify the reverse chain from destination to candidate (as in the previous heuristic).
    # 2. For each edge in the reverse chain, check if the current edge (current_node -> candidate) 
    #    would cross it using the triangle inequality criterion.
    # 3. Sum the penalties for likely crossings.
    
    # This avoids the need for explicit coordinates!
    
    def estimate_remaining_cost_reverse_and_crossing_penalty(candidate, remaining_unvisited, dest_node, dist_matrix):
        """
        Estimates the cost to go from `candidate` to `dest_node` visiting all nodes in `remaining_unvisited`.
        Also returns a crossing penalty for the edge (current_node -> candidate).
        
        Returns:
        (total_cost, crossing_penalty)
        """
        if len(remaining_unvisited) == 0:
            return dist_matrix[candidate, dest_node], 0.0
        
        temp_unvisited = list(remaining_unvisited)
        current_pos = dest_node
        total_cost = 0.0
        crossing_penalty = 0.0
        
        # Store the reverse chain to check for crossings against the current edge
        # The reverse chain is built as: dest_node -> r1 -> r2 -> ... -> rk
        # The forward path segment will be: candidate -> rk -> ... -> r1 -> dest_node
        
        # We will build the list of nodes in the reverse chain order (starting from dest)
        reverse_chain = [dest_node]
        
        while len(temp_unvisited) > 0:
            candidates = temp_unvisited
            dists = dist_matrix[current_pos, candidates]
            
            min_idx = np.argmin(dists)
            next_node = candidates[min_idx]
            
            # Add cost of this leg
            total_cost += dists[min_idx]
            
            # Move current pos to this node
            current_pos = next_node
            
            # Add to reverse chain
            reverse_chain.append(current_pos)
            
            # Remove visited node
            temp_unvisited.pop(min_idx)
            
        # Final connection from candidate to the end of the reverse chain (rk)
        final_edge_to = current_pos
        total_cost += dist_matrix[candidate, final_edge_to]
        
        # Now, estimate crossing penalty for the edge E_current = (current_node, candidate)
        # We check this edge against all edges in the reverse chain path.
        # The reverse chain path edges are (reverse_chain[i], reverse_chain[i+1])
        
        # To check if edge (A, B) crosses (C, D) in Euclidean space without coordinates:
        # They cross if dist(A, C) + dist(B, D) < dist(A, B) + dist(C, D) AND
        #             dist(A, D) + dist(B, C) < dist(A, B) + dist(C, D)
        # Actually, a simpler necessary condition for intersection in convex hulls etc is complex.
        # But a common heuristic is: if the sum of the cross diagonals is less than the sum of the parallel sides,
        # it suggests a crossing.
        # Specifically, for segments AB and CD:
        # If dist(A,C) + dist(B,D) < dist(A,B) + dist(C,D) then they might cross.
        # We can assign a penalty proportional to the "depth" of this inequality.
        
        # Note: current_node is not in reverse_chain, and candidate is the end of the current edge.
        # The reverse chain connects dest to final_edge_to (which is the last node visited in reverse greedy).
        # The edge (candidate, final_edge_to) is also part of the future path.
        
        # Let's define the future edges as:
        # 1. The edges forming the reverse chain: (dest, r1), (r1, r2), ..., (r_{k-1}, rk)
        # 2. The edge (candidate, rk)
        
        # We check the edge (current_node, candidate) against all these future edges.
        
        future_edges = []
        # Edges in the chain
        for i in range(len(reverse_chain) - 1):
            future_edges.append((reverse_chain[i], reverse_chain[i+1]))
        # Edge from candidate to the last node in reverse chain
        future_edges.append((candidate, final_edge_to))
        
        # Current edge
        A, B = current_node, candidate
        
        # Precompute dist(A, B)
        dist_AB = dist_matrix[A, B]
        
        for C, D in future_edges:
            # Skip if edges share a node (they connect, not cross)
            if A == C or A == D or B == C or B == D:
                continue
                
            dist_AC = dist_matrix[A, C]
            dist_BD = dist_matrix[B, D]
            dist_AD = dist_matrix[A, D]
            dist_BC = dist_matrix[B, C]
            dist_CD = dist_matrix[C, D]
            
            # Heuristic: If the cross sum is significantly smaller than the straight sum, 
            # it implies a crossing in Euclidean geometry.
            # Penalty is added if cross_sum < straight_sum
            
            cross_sum_1 = dist_AC + dist_BD
            cross_sum_2 = dist_AD + dist_BC
            straight_sum = dist_AB + dist_CD
            
            # If either cross sum is smaller than straight sum, it's a potential crossing
            # The "penalty" can be the difference.
            penalty_1 = max(0, straight_sum - cross_sum_1)
            penalty_2 = max(0, straight_sum - cross_sum_2)
            
            # Take the max penalty as the measure of crossing likelihood
            crossing_penalty += max(penalty_1, penalty_2)
            
        return total_cost, crossing_penalty

    best_candidate = -1
    min_total_estimated_cost = float('inf')
    
    # Iterate over all unvisited nodes to find the best one
    for i in range(n_unvisited):
        candidate = unvisited_nodes[i]
        
        # 1. Cost to move from current_node to candidate
        step_cost = distance_matrix[current_node, candidate]
        
        # 2. Remaining unvisited nodes after visiting candidate
        remaining_nodes = unvisited_nodes[unvisited_nodes != candidate]
        
        # 3. Estimate remaining tour cost and crossing penalty
        remaining_est_cost, crossing_penalty = estimate_remaining_cost_reverse_and_crossing_penalty(
            candidate=candidate,
            remaining_unvisited=remaining_nodes,
            dest_node=destination_node,
            dist_matrix=distance_matrix
        )
        
        # 4. Total estimated cost: step + remaining + penalty
        # The penalty increases the cost, thus discouraging edges that cause crossings.
        total_estimated_cost = step_cost + remaining_est_cost + crossing_penalty
        
        if total_estimated_cost < min_total_estimated_cost:
            min_total_estimated_cost = total_estimated_cost
            best_candidate = candidate
            
    return int(best_candidate)

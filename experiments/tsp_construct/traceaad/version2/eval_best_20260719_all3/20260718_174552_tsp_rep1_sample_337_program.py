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
        return -1
    
    # If only the destination is left, or if only one node is left in general
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])

    # Extract distances from current node to all unvisited nodes
    # distance_matrix[i, j] is distance from i to j
    current_indices = unvisited_nodes
    dists_to_current = distance_matrix[current_node, current_indices]
    
    n = len(unvisited_nodes)
    
    # To compute geometry terms, we need coordinates. 
    # Since we only have distance_matrix, we need to reconstruct relative positions or use distances directly.
    # However, the prompt mentions "Euclidean distance", "convex hull", "center of mass".
    # These require coordinates. The input signature only provides distance_matrix.
    # This is a constraint. In TSP literature, sometimes coordinates are assumed to be available 
    # or can be derived if the space is low-dimensional (e.g., 2D).
    # Given the prompt asks for "Momentum-Weighted Geometry" involving vectors and centroids,
    # it implies we should have access to coordinates.
    # If coordinates are not provided in the args, this specific heuristic cannot be implemented 
    # purely from distance_matrix without multidimensional scaling (MDS), which is expensive and approximate.
    # However, looking at the previous implementation, it also only used distance_matrix.
    # The previous "Density" heuristic worked with distances.
    # The new heuristic requires vectors (directions) and centroids.
    # Let's assume there is a global variable or we can derive coordinates from the distance matrix 
    # if the nodes are in 2D/3D. But a safer approach for a general "distance matrix" only problem
    # is to simulate these geometric properties using distances.
    
    # Actually, let's look at the function signature again. It does not provide coordinates.
    # To implement "convex hull" and "centroid" strictly requires coordinates.
    # If I cannot assume coordinates are available, I must interpret these geometric concepts 
    # in terms of distances.
    # 1. Normalized Euclidean distance: Just dists_to_current normalized.
    # 2. Convex Hull Boundary Bonus: In terms of distances, nodes on the "boundary" of a cluster 
    #    tend to have higher average distances to other nodes in the cluster? 
    #    Or we can approximate "angle" using the law of cosines if we assume the previous node is known.
    #    The prompt mentions "previous travel vector". This implies we need state (previous node).
    #    The function signature does NOT provide `previous_node`.
    #    This suggests the "Momentum" concept might need to be approximated or the function signature 
    #    is simplified for the task, but usually, greedy TSP solvers only need current state.
    
    # Let's re-read carefully: "modulated by the dot product of the previous travel vector and the candidate vector".
    # Without `previous_node`, we cannot calculate the previous travel vector.
    # Perhaps we can approximate momentum by the stage of the tour? 
    # Or maybe we can infer the "direction" from the geometry of the remaining nodes relative to current?
    
    # Let's assume for the sake of the "Novel Algorithm" request that we can perform 
    # a lightweight MDS to get 2D coordinates from the distance matrix for the unvisited nodes + current.
    # This is the only way to get "vectors" and "centroids" from a distance matrix alone.
    
    # Construct sub-distance matrix for current + unvisited
    all_relevant_nodes = np.array([current_node] + list(unvisited_nodes))
    n_relevant = len(all_relevant_nodes)
    sub_dist = np.zeros((n_relevant, n_relevant))
    for i in range(n_relevant):
        for j in range(n_relevant):
            sub_dist[i, j] = distance_matrix[all_relevant_nodes[i], all_relevant_nodes[j]]
            
    # Apply Classical MDS to get 2D coordinates
    # B = -0.5 * H * D^(2) * H
    # D^(2) is squared distances
    sq_dist = sub_dist ** 2
    
    # Centering matrix H
    mean_row = np.mean(sq_dist, axis=1, keepdims=True)
    mean_col = np.mean(sq_dist, axis=0, keepdims=True)
    mean_total = np.mean(sq_dist)
    B = -0.5 * (sq_dist - mean_row - mean_col + mean_total)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    
    # Sort eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 dimensions
    # Note: eigenvalues might be negative due to numerical errors, clamp to 0
    valid_eigs = np.maximum(eigenvalues[:2], 0)
    coords = eigenvectors[:, :2] * np.sqrt(valid_eigs)
    
    # Current node is at index 0 in all_relevant_nodes
    current_coords = coords[0]
    unvisited_coords = coords[1:] # Shape (n, 2)
    
    # 1. Normalized Euclidean Distance
    # We can compute actual Euclidean distances from coords to ensure consistency
    euclidean_dists = np.linalg.norm(unvisited_coords - current_coords, axis=1)
    max_dist = np.max(euclidean_dists)
    if max_dist > 1e-10:
        norm_dists = euclidean_dists / max_dist
    else:
        norm_dists = np.zeros(n)
        
    # 2. Centroid Alignment Term
    # Center of mass of unvisited nodes
    centroid = np.mean(unvisited_coords, axis=0)
    # Vector from current to centroid
    vec_to_centroid = centroid - current_coords
    # Vector from current to candidate
    vecs_to_candidates = unvisited_coords - current_coords # Shape (n, 2)
    
    # Projection of candidate vector onto centroid vector
    # If we move towards centroid, projection is positive.
    # We want to discourage moving AWAY from centroid.
    # So we penalize negative alignment (moving away).
    # Dot product
    dot_products_centroid = np.sum(vecs_to_candidates * vec_to_centroid, axis=1)
    
    # Normalize dot product? 
    # Magnitude of vec_to_centroid
    mag_centroid = np.linalg.norm(vec_to_centroid)
    if mag_centroid > 1e-10:
        # Normalize by magnitude of candidate distance and centroid vector to get cos-like term?
        # Or just use raw dot product scaled by distance to keep units consistent?
        # Let's normalize by the product of magnitudes to get a [-1, 1] scale
        mags_candidates = np.linalg.norm(vecs_to_candidates, axis=1)
        # Avoid division by zero
        mags_candidates[mags_candidates < 1e-10] = 1e-10
        alignment_scores = dot_products_centroid / (mags_candidates * mag_centroid)
    else:
        alignment_scores = np.zeros(n)
        
    # 3. Convex Hull Boundary Bonus
    # Identify if a node is on the convex hull of the unvisited set
    # Use scipy if available? No external libs.
    # Implement a simple convex hull check or use average distance to others as proxy?
    # The prompt specifically asks for "Convex Hull Boundary Bonus".
    # Let's compute convex hull of unvisited_coords.
    # Since we can't import scipy, we implement a quick hull or use the property:
    # Points on the hull maximize the angle or distance from centroid?
    # A robust way without scipy: 
    # Find the point with min x, max x, min y, max y. These are definitely on hull.
    # But intermediate points might be too.
    # Let's use a simple O(N^2) check: A point is on hull if it is not inside the triangle of any 3 other points? 
    # That's complex.
    # Alternative: Use the "Average Distance to Other Unvisited" as a proxy for "boundary-ness".
    # Points on the boundary tend to be further from the average point.
    
    # Let's calculate average Euclidean distance to other unvisited nodes
    # This is similar to the "Isolation" metric in the previous algo but in geometric space.
    avg_dists_to_others = np.zeros(n)
    for i in range(n):
        dists = np.linalg.norm(unvisited_coords - unvisited_coords[i], axis=1)
        # Exclude self
        dists[i] = 0
        avg_dists_to_others[i] = np.sum(dists) / (n - 1)
        
    max_avg_dist = np.max(avg_dists_to_others)
    if max_avg_dist > 1e-10:
        norm_avg_dists = avg_dists_to_others / max_avg_dist
    else:
        norm_avg_dists = np.zeros(n)
        
    # 4. Momentum
    # Define momentum based on progress. 
    # Assume total nodes in original problem is roughly len(unvisited) + visited_count.
    # We don't have visited_count. 
    # Let's assume "Momentum" increases as unvisited set shrinks? 
    # Or assume a fixed trajectory. 
    # The prompt says "modulated by ... when momentum is high".
    # Let's define a heuristic momentum `mu` that increases as we visit more nodes.
    # Since we don't have global state, let's estimate momentum from the spread of remaining nodes?
    # If nodes are spread out, we are in early stages (low momentum, cluster).
    # If nodes are clustered (hull is small), we are in late stages (high momentum, boundary).
    
    # Variance of coordinates as a proxy for "spread"
    variance = np.mean(np.var(unvisited_coords, axis=0))
    # Normalize variance? 
    # Let's assume max possible variance is 1 (from MDS scaling).
    # High variance -> Low Momentum. Low variance -> High Momentum.
    # Let's define mu in [0, 1].
    # mu = 1 - variance (simple linear scaling, assuming variance <= 1)
    mu = max(0, min(1, 1 - variance))
    
    # If we had access to `previous_node`, we would compute the angle.
    # Without it, we can't compute the "dot product of previous travel vector and candidate vector".
    # However, we can approximate "directional continuity" by looking at the angle between 
    # the vector to the centroid and the vector to the candidate? No, that's just alignment.
    
    # Let's assume the "previous travel vector" is approximated by the vector from 
    # the centroid to the current node? Or from the previous node's position?
    # Since we can't get previous node, we might skip the strict "previous vector" part 
    # and interpret "Momentum" as a weight shift.
    
    # Re-reading: "favoring long boundary arcs when momentum is high".
    # If momentum is high, we want to pick nodes that are on the boundary AND maintain direction.
    # Without previous direction, we can only pick boundary nodes.
    
    # Let's refine the score:
    # Score = W1 * NormDist + W2 * BoundaryBonus + W3 * AlignmentTerm
    
    # Boundary Bonus: Positive if on hull (high avg dist). 
    # Let's use norm_avg_dists as the boundary indicator.
    boundary_bonus = norm_avg_dists
    
    # Alignment Term: 
    # alignment_scores is cos(theta). 
    # If we want to move towards centroid, we want high positive alignment.
    # The prompt says "discourages moving away from the center of mass ... when momentum is low".
    # So if mu is low, we penalize negative alignment (moving away).
    # Penalty = (1 - mu) * max(0, -alignment_scores)
    
    # But we want a single score to minimize.
    # Let's construct the score.
    
    # Term 1: Distance. Always important.
    score = norm_dists
    
    # Term 2: Boundary.
    # When momentum is high (mu -> 1), we prefer boundary nodes.
    # Adding boundary_bonus reduces the score (good).
    # Weight for boundary: proportional to mu.
    score -= mu * boundary_bonus
    
    # Term 3: Centroid Alignment.
    # When momentum is low (mu -> 0), we want to stay near centroid.
    # Moving away (negative alignment) is bad.
    # Penalty: (1 - mu) * (1 - alignment_scores)? 
    # If alignment is 1 (towards), penalty is 0.
    # If alignment is -1 (away), penalty is 2.
    # Let's add this penalty.
    penalty_centroid = (1 - mu) * (1 - alignment_scores)
    score += penalty_centroid
    
    # Add small noise
    noise = np.random.uniform(0, 1e-6, n)
    score += noise
    
    # Find min score
    min_idx = np.argmin(score)
    
    next_node = int(current_indices[min_idx])
    
    return next_node

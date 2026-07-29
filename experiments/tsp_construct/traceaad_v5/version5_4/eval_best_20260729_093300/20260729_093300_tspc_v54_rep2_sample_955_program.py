import numpy as np
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.manifold import MDS
def _compute_angles_mds_anchored(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Computes direction scores using vector angles derived from MDS coordinates.
    Anchors current_node at origin and destination_node at (d, 0) to stabilize direction.
    """
    all_nodes = np.concatenate([[current_node, destination_node], unvisited_nodes])
    # Ensure unique nodes for MDS
    unique_nodes = np.unique(all_nodes)
    
    # Subset distance matrix for relevant nodes
    sub_dist = distance_matrix[np.ix_(unique_nodes, unique_nodes)]
    n_nodes = len(unique_nodes)
    
    # Map node IDs to indices in the unique list
    node_to_idx = {node: i for i, node in enumerate(unique_nodes)}
    
    idx_curr = node_to_idx[current_node]
    idx_dest = node_to_idx[destination_node]
    
    # Distance between current and destination in the metric space
    dist_curr_dest = sub_dist[idx_curr, idx_dest]
    
    # We want to embed these n_nodes points such that:
    # Point idx_curr is at (0, 0)
    # Point idx_dest is at (dist_curr_dest, 0)
    # However, sklearn's MDS does not support fixed anchors directly.
    # Strategy: 
    # 1. If n_nodes <= 2, we handle manually.
    # 2. For larger n, we can't fix anchors easily in sklearn MDS without custom optimization.
    #    But we can approximate by ensuring the first two points in the input matrix 
    #    are current and destination, and then rotating/translating the result.
    
    # Reorder unique_nodes so current is first, destination is second
    ordered_nodes = [current_node, destination_node]
    for n in unique_nodes:
        if n != current_node and n != destination_node:
            ordered_nodes.append(n)
    
    ordered_nodes = np.array(ordered_nodes)
    # Reorder distance matrix
    # Find indices of ordered nodes in unique_nodes
    ordered_indices = [node_to_idx[n] for n in ordered_nodes]
    sub_dist_ordered = sub_dist[np.ix_(ordered_indices, ordered_indices)]
    
    try:
        # Use MDS to embed
        mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto', random_state=42)
        coords = mds.fit_transform(sub_dist_ordered)
        
        # coords[0] is embedding of current_node
        # coords[1] is embedding of destination_node
        
        # Translate so current_node is at origin
        coords = coords - coords[0]
        
        # Rotate so destination_node lies on positive X-axis
        dest_coord = coords[1]
        angle = np.arctan2(dest_coord[1], dest_coord[0])
        
        # Rotation matrix
        rot = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)]
        ])
        
        coords = coords @ rot.T
        
        # Now coords[0] is approx (0,0) and coords[1] is approx (dist, 0)
        
    except Exception as e:
        # Fallback if MDS fails
        coords = np.zeros((len(ordered_nodes), 2))
        # Simple 1D projection along the vector to destination isn't possible without 2D
        # Just place current at 0, dest at d, others at 0
        coords[1] = [dist_curr_dest, 0]
        
    # Map back to node IDs for all unvisited nodes
    node_to_coord = {node: coord for node, coord in zip(ordered_nodes, coords)}
    
    # Vector from current to destination in the embedded space
    # It should be aligned with X-axis now
    vec_curr_dest = node_to_coord[destination_node] - node_to_coord[current_node]
    
    cos_theta_vals = []
    for node in unvisited_nodes:
        vec_curr_cand = node_to_coord[node] - node_to_coord[current_node]
        
        # Compute cosine of angle
        dot_product = np.dot(vec_curr_dest, vec_curr_cand)
        norm_dest = np.linalg.norm(vec_curr_dest)
        norm_cand = np.linalg.norm(vec_curr_cand)
        
        if norm_dest == 0 or norm_cand == 0:
            cos_theta = 0.0
        else:
            cos_theta = dot_product / (norm_dest * norm_cand)
            
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        cos_theta_vals.append(cos_theta)
        
    return np.array(cos_theta_vals)


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
    # Handle edge case: no unvisited nodes
    if len(unvisited_nodes) == 0:
        return destination_node
    
    # Handle edge case: only one unvisited node
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    n_unvisited = len(unvisited_nodes)
    
    # Precompute sub-matrix for unvisited nodes to optimize distance lookups
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    all_distances = sub_matrix.flatten()
    avg_dist_global = np.mean(all_distances) if len(all_distances) > 0 else 1.0
    avg_dist_global = max(avg_dist_global, 1e-10)
    
    # Calculate distances from current node to each unvisited candidate
    current_distances = distance_matrix[current_node, unvisited_nodes]
    
    # Calculate average distance from each candidate to other unvisited nodes (Isolation Heuristic)
    sum_dists_to_others = np.sum(sub_matrix, axis=1)
    avg_dists_to_others = sum_dists_to_others / (n_unvisited - 1)
    
    # Forward Progress Heuristic (MDS-based Vector Angle):
    # Compute true cosine similarity using reconstructed 2D coordinates from MDS
    # Using anchored MDS for stability
    cos_theta = _compute_angles_mds_anchored(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Direction score: Penalize low cosine similarity (large angles)
    # If theta = 0 (straight towards dest), cos(theta) = 1, score = 0.
    direction_score = 1.0 - cos_theta
    
    # Dynamic Edge-Length Normalization using Percentile (75th)
    if len(all_distances) > 0:
        percentile_edge_length = np.percentile(all_distances, 75)
    else:
        percentile_edge_length = avg_dist_global
    percentile_edge_length = max(percentile_edge_length, 1e-10)

    # Urgency weighting for forward progress heuristic
    dist_cand_dest = distance_matrix[unvisited_nodes, destination_node]
    dist_curr_dest = distance_matrix[current_node, destination_node]
    avg_unvisited_dist_to_dest = np.mean(dist_cand_dest)
    
    epsilon = 1e-8
    # Ratio of current distance to destination vs average unvisited distance
    urgency_ratio = dist_curr_dest / (avg_unvisited_dist_to_dest + epsilon)
    
    # Cap urgency weight to prevent explosion
    capped_urgency = np.minimum(urgency_ratio, 3.0)

    future_costs = np.zeros(n_unvisited)
    
    for i, candidate in enumerate(unvisited_nodes):
        immediate_dist = current_distances[i]
        
        # Isolation term
        isolation_score = immediate_dist / (avg_dists_to_others[i] + epsilon)
        
        # O(N log N) Approximate MST Estimation for Future Costs
        remaining = np.delete(unvisited_nodes, i)
        
        if len(remaining) == 0:
            future_cost = distance_matrix[candidate, destination_node]
        else:
            rem_indices = remaining
            # Extract submatrix for remaining nodes
            sub_matrix_rem = distance_matrix[np.ix_(rem_indices, rem_indices)]
            
            # Build sparse matrix for MST calculation
            # Use csr_matrix for efficiency
            sparse_rem = csr_matrix(sub_matrix_rem)
            
            try:
                # Compute MST using scipy
                mst = minimum_spanning_tree(sparse_rem)
                mst_cost = mst.sum()
            except:
                # Fallback if scipy fails or returns unexpected result
                mst_cost = 0.0
            
            # Min distance from candidate to any remaining unvisited node
            dists_to_remaining = distance_matrix[candidate, remaining]
            min_dist_to_remaining = np.min(dists_to_remaining)
            
            # Average distance from remaining unvisited nodes to destination
            avg_dist_to_dest = np.mean(distance_matrix[remaining, destination_node])
            
            # Heuristic: MST cost of remaining + Min edge to connect to next + Avg cost to finish
            future_cost = mst_cost + min_dist_to_remaining + avg_dist_to_dest
        
        normalized_future = future_cost / avg_dist_global
        
        # Future weight scales with number of unvisited nodes
        future_weight = 0.3 + 0.4 * (n_unvisited / max(n_unvisited, 1))
        
        # Base score
        score = isolation_score + future_weight * normalized_future
        
        # Dynamic Edge-Length Modulation for Forward Progress
        normalized_edge_len = immediate_dist / percentile_edge_length
        
        # Base direction weight depends on number of nodes remaining
        base_direction_weight = 0.3 * (1.0 / (n_unvisited + 1))
        
        # Urgency-weighted direction weight
        direction_weight = base_direction_weight * (1.0 + 0.5 * normalized_edge_len) * capped_urgency
        
        score += direction_weight * direction_score[i]
        
        # Isolation-Modulated Jump Penalty
        # Integrate primary program's logic: Scale penalty by isolation factor
        jump_ratio = immediate_dist / avg_dist_global
        
        # Use fixed ratio-based threshold (2.0 * global_avg_dist) as stabilized in primary history
        if jump_ratio > 2.0:
            base_jump_penalty = (jump_ratio - 2.0) * avg_dist_global * 0.5
            
            # Isolation Sensitivity Modulation from primary program
            # Scale penalty by (1.0 + 0.5 * (avg_dists_to_others[i] / avg_dist_global))
            isolation_factor = avg_dists_to_others[i] / avg_dist_global
            jump_penalty = base_jump_penalty * (1.0 + 0.5 * isolation_factor)
            
            score += jump_penalty
        
        future_costs[i] = score
    
    best_idx = np.argmin(future_costs)
    return int(unvisited_nodes[best_idx])

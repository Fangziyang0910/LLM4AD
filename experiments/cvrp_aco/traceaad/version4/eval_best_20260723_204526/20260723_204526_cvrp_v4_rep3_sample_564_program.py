import numpy as np

def heuristics(distance_matrix: np.ndarray, coordinates: np.ndarray, demands: np.ndarray, capacity: int) -> np.ndarray:
    """Return edge desirability values for CVRP ant colony optimization.

    Args:
        distance_matrix: Pairwise Euclidean distances with shape (n, n).
        coordinates: Node coordinates with shape (n, 2). Node 0 is the depot.
        demands: Node demands with shape (n,). The depot demand is zero.
        capacity: Capacity shared by all vehicles.

    Returns:
        An (n, n) edge-prior matrix. Larger values make an edge more likely
        to be sampled. Values at or below zero are treated as 1e-9.
    """
    n = distance_matrix.shape[0]
    
    # Handle edge case
    if n < 2:
        return np.ones((n, n))

    # Depot is at index 0
    depot_coords = coordinates[0:1]  # Shape (1, 2)
    
    # Calculate angles of all nodes relative to depot
    # Vector from depot to each node
    vectors = coordinates - depot_coords  # Shape (n, 2)
    
    # Angle in radians
    angles = np.arctan2(vectors[:, 1], vectors[:, 0]) # Shape (n,)
    
    # Normalize angles to [0, 2pi) for easier diff calculation
    angles = angles % (2 * np.pi)
    
    # 1. Angular Alignment Component (Base)
    # Compute pairwise absolute difference in angles
    diff_matrix = angles[:, np.newaxis] - angles[np.newaxis, :] # Shape (n, n)
    
    # Wrap differences to [-pi, pi]
    diff_matrix = np.mod(diff_matrix + np.pi, 2 * np.pi) - np.pi
    
    # Absolute angular difference
    abs_angle_diff = np.abs(diff_matrix) # Shape (n, n)
    
    # Convert angle difference to a score: 
    # Smaller diff -> higher score. 
    angle_scale = 1.0 
    score_angle = np.exp(-angle_scale * abs_angle_diff)
    
    # 2. Cluster-Centric Angular Bias
    # Identify K-nearest neighbors for each node to define local cluster centers.
    # We use K=6 as a reasonable default for local density.
    K = min(6, n - 1)
    
    # Get indices of K nearest neighbors for each node (excluding self)
    knn_indices = np.argsort(distance_matrix, axis=1)[:, 1:K+1] # Shape (n, K)
    
    # Get coordinates of KNNs
    k_coords = coordinates[knn_indices] # Shape (n, K, 2)
    
    # Compute cluster centers for each node (average of KNN coordinates)
    cluster_centers = np.mean(k_coords, axis=1) # Shape (n, 2)
    
    # Vector from depot to cluster centers
    cluster_vectors = cluster_centers - depot_coords # Shape (n, 2)
    
    # Angles of cluster centers
    cluster_angles = np.arctan2(cluster_vectors[:, 1], cluster_vectors[:, 0]) # Shape (n,)
    cluster_angles = cluster_angles % (2 * np.pi)
    
    # Compute angular difference between node i's cluster center and node j's cluster center
    # Angular similarity between node j and the cluster center of node i.
    
    # Angle of cluster center of i
    angle_i_cluster = cluster_angles # Shape (n,)
    
    # Difference between node j's angle and node i's cluster center angle
    diff_cluster = angles[np.newaxis, :] - angle_i_cluster[:, np.newaxis] # Shape (n, n)
    diff_cluster = np.mod(diff_cluster + np.pi, 2 * np.pi) - np.pi
    abs_cluster_diff = np.abs(diff_cluster)
    
    # Bias factor: exp(-scale * diff)
    cluster_bias_scale = 2.0 # Stronger bias towards cluster centers
    cluster_bias = np.exp(-cluster_bias_scale * abs_cluster_diff)
    
    # Combine with base angular score
    score_angle_cluster = score_angle * cluster_bias

    # 3. Demand-to-Distance Ratio Component
    demands_dest = demands[np.newaxis, :] # Shape (1, n) -> (n, n)
    epsilon = 1e-9
    dist_safe = distance_matrix + epsilon
    
    demand_dist_ratio = demands_dest / dist_safe
    
    max_ratio = np.max(demand_dist_ratio)
    if max_ratio > 0:
        score_demand_dist = demand_dist_ratio / max_ratio
    else:
        score_demand_dist = np.ones_like(demand_dist_ratio) * 1e-9

    # 4. Dynamic Capacity Feasibility Component
    # Calculate average demand per unvisited node (excluding depot)
    customer_demands = demands[1:]
    avg_demand = np.mean(customer_demands) if len(customer_demands) > 0 else 0
    
    # Dynamic threshold: Use average demand as a baseline for "standard" load
    cap_safe = max(capacity, 1e-9)
    dynamic_threshold = avg_demand / cap_safe
    dynamic_threshold = max(dynamic_threshold, 1e-6)
    
    demands_src = demands[:, np.newaxis] # Shape (n, n)
    used_cap = demands_src + demands_dest
    used_cap_ratio = used_cap / cap_safe
    
    # Scale alpha_cap inversely with the ratio of capacity to mean customer demand
    # This ratio represents the approximate number of average-demand customers a vehicle can hold.
    # Higher ratio (low density) -> lower alpha (less aggressive penalty)
    # Lower ratio (high density) -> higher alpha (more aggressive penalty)
    if avg_demand > 0:
        scale_factor = capacity / avg_demand
    else:
        scale_factor = 1.0
    
    # Base alpha scaled by the inverse of the number of customers fitting in capacity
    # We use a base alpha of 10.0, divided by the scale factor.
    # If capacity=100, avg_demand=10 -> scale=10 -> alpha=1.0 (mild)
    # If capacity=100, avg_demand=50 -> scale=2 -> alpha=5.0 (moderate)
    # If capacity=100, avg_demand=100 -> scale=1 -> alpha=10.0 (strict)
    base_alpha = 10.0
    alpha_cap = base_alpha / max(scale_factor, 1e-9)
    
    # Soft threshold penalty based on dynamic threshold
    over_use = np.maximum(0, used_cap_ratio - dynamic_threshold)
    cap_penalty = np.exp(-alpha_cap * over_use)
    
    # 5. Sweep Angle Consistency Penalty
    # We approximate the "incoming" angle at node i by the angle of the vector from Depot to i.
    # We compare this with the "outgoing" angle from i to j.
    # A consistent sweep implies these angles should be similar.
    
    # Angle from i to j
    vectors_ij = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :] # Shape (n, n, 2)
    angles_ij = np.arctan2(vectors_ij[:, :, 1], vectors_ij[:, :, 0]) # Shape (n, n)
    
    # Angle from Depot to i (Incoming approximation)
    angle_depot_to_i = angles[:, np.newaxis] # Shape (n, 1) -> broadcast to (n, n)
    
    # Difference between outgoing angle and incoming angle
    diff_sweep = angles_ij - angle_depot_to_i
    diff_sweep = np.mod(diff_sweep + np.pi, 2 * np.pi) - np.pi
    abs_sweep_diff = np.abs(diff_sweep)
    
    # Penalty: exp(-scale * abs_diff)
    # If diff is small (consistent sweep), score is high (~1).
    # If diff is large (sharp turn/reversal), score is low.
    sweep_scale = 1.5
    sweep_consistency = np.exp(-sweep_scale * abs_sweep_diff)
    
    # Apply penalty to non-depot edges. For depot returns, we don't apply this 
    # because the "next" node is fixed (depot) and the concept of sweep consistency
    # changes (it's a return). We let the depot bias handle it.
    sweep_mask = np.ones_like(sweep_consistency)
    sweep_mask[:, 0] = 1.0 # Neutralize penalty for edges going to depot
    
    score_sweep = sweep_consistency * sweep_mask

    # 6. Combine Scores
    heuristic_matrix = score_angle_cluster * score_demand_dist * cap_penalty * score_sweep

    # 7. Final Distance Penalty
    distance_penalty = 1.0 / dist_safe
    heuristic_matrix *= distance_penalty
    
    # 8. Depot Proximity Bias with Refined Capacity and Customer Count Scaling
    # Calculate distances from each node to the depot (column 0 of distance matrix)
    dist_from_depot = distance_matrix[:, 0]
    max_dist = np.max(dist_from_depot[1:]) # Ignore depot itself for max
    if max_dist <= 0:
        max_dist = 1.0 # Avoid division by zero
        
    # Normalized distance component
    dist_norm = dist_from_depot / max_dist
    
    # Refined logic:
    # 1. Remaining Capacity Ratio: 1 - (demand_src / capacity)
    #    High remaining cap -> Low pressure to return.
    #    Low remaining cap -> High pressure to return.
    remaining_cap_ratio = 1.0 - (demands / cap_safe)
    
    # 2. Scale exponential decay inversely with number of unvisited customers.
    #    As customers are visited (simulated by total unvisited count), pressure increases.
    #    We estimate "unvisited" roughly by the total number of nodes n. 
    #    In a static heuristic, we can't know exact unvisited, but we can scale the 
    #    intensity of the depot return based on global problem density.
    #    Alternatively, the prompt implies a dynamic feel. Since this is static, 
    #    we assume the "pressure" should be higher when the route is "fuller".
    #    Since we don't know the route, we use the demand/capacity ratio as a proxy for fullness.
    #    However, the prompt specifically asks to scale inversely with unvisited customers.
    #    In a static matrix, "unvisited" is constant (n-1). 
    #    Perhaps the intent is to make the boost sensitive to the *potential* route length?
    #    Let's stick to the explicit request: scale exp factor inversely with unvisited.
    #    Let's assume "unvisited" ~ n. If n is large, routes can be longer, so depot pressure 
    #    per step should be lower? Or higher?
    #    Actually, usually depot pressure is about finishing the *current* route.
    #    Let's interpret "unvisited customers" in the context of the heuristic generation 
    #    as a global scaling factor. If there are many customers, we might need many routes.
    #    Let's use a fixed scaling factor based on n to normalize the exponential growth.
    
    num_customers = n - 1
    if num_customers > 0:
        # Scale factor: inversely proportional to number of customers
        # If many customers, scale down the exponent slightly to allow longer routes?
        # Or scale up? If we scale inversely, 1/n is small.
        # Let's use a base beta and divide by a function of n.
        beta_base = 5.0
        beta_depot = beta_base / np.sqrt(num_customers)
    else:
        beta_depot = 1.0
        
    # Combined boost factor: exponential of (beta * (remaining_cap_inverse + dist_ratio))
    # Remaining cap ratio is 1 - load. So (1 - remaining) is load.
    # We want high pressure when remaining is LOW.
    # So we use (1 - remaining_cap_ratio) = demands/capacity
    
    load_ratio = demands / cap_safe
    
    # The boost applies to the urgency of returning. 
    # High load + Far distance = Strong urge to close route.
    depot_boost = np.exp(beta_depot * (load_ratio + dist_norm))
    
    # Apply boost only to edges going to depot (column 0)
    heuristic_matrix[:, 0] *= depot_boost

    # 9. Remote Intra-Cluster Bonus
    # Boosts edges between nodes that are both far from the depot and close to each other.
    # This encourages servicing remote clusters together to minimize deadhead.
    
    # Get distances from depot for source and destination nodes
    dist_src_depot = dist_from_depot[:, np.newaxis] # Shape (n, 1)
    dist_dst_depot = dist_from_depot[np.newaxis, :] # Shape (1, n)
    
    # Normalize distances to [0, 1]
    norm_dist_src = dist_src_depot / max_dist
    norm_dist_dst = dist_dst_depot / max_dist
    
    # Calculate normalized inter-node distance
    # We want small inter-node distance to be high value
    # max_pairwise_dist = np.max(distance_matrix[1:, 1:]) # Avoid depot
    # For simplicity and robustness, use max_dist as a proxy scale or np.max(distance_matrix)
    max_pairwise_dist = np.max(distance_matrix)
    if max_pairwise_dist <= 0:
        max_pairwise_dist = 1.0
        
    norm_inter_dist = distance_matrix / max_pairwise_dist
    
    # Bonus factor: High when both nodes are far (high norm_dist) AND close to each other (low norm_inter_dist)
    # We use an exponential form: exp(k * (avg_norm_dist - norm_inter_dist))
    # avg_norm_dist: average distance of the pair from depot
    avg_norm_dist = (norm_dist_src + norm_dist_dst) / 2.0
    
    # Difference: Positive if they are far and close to each other
    remote_close_diff = avg_norm_dist - norm_inter_dist
    
    # Adjust k_bonus to be inversely proportional to median customer distance from depot
    # This adapts the boost intensity to the problem's geometric spread.
    median_dist_from_depot = np.median(dist_from_depot[1:]) if n > 1 else 1.0
    if median_dist_from_depot > 0:
        # Base k value scaled by median distance.
        # Higher median distance (spread out problem) -> lower k_bonus (less aggressive bonus relative to scale)
        # Lower median distance (dense problem) -> higher k_bonus (more aggressive bonus)
        # We normalize by a base scale to keep magnitudes reasonable.
        base_k = 10.0
        # Use a reference scale to normalize the median distance if necessary, 
        # but direct inverse proportionality is requested.
        k_bonus = base_k / median_dist_from_depot
    else:
        k_bonus = 2.0 # Fallback
        
    # Apply bonus only if diff is positive (i.e., they are closer to each other than to the depot on average)
    # And scale it to be significant but not overwhelming
    remote_bonus = np.exp(k_bonus * np.maximum(0, remote_close_diff))
    
    # Do not apply to depot edges (column 0 or row 0)
    depot_mask = np.ones_like(remote_bonus)
    depot_mask[:, 0] = 1.0
    depot_mask[0, :] = 1.0
    
    heuristic_matrix *= remote_bonus * depot_mask

    # 10. Geometric Cluster Proximity Bonus
    # Boosts edges between nodes that share many K-nearest neighbors.
    # This indicates they belong to the same dense geometric region.
    
    # KNN indices already computed in Step 2: knn_indices (n, K)
    # We need to count overlaps between row i and row j of knn_indices.
    
    # Efficient overlap calculation using broadcasting or loops?
    # n is typically small enough for n*K*K or n^2*K operations.
    # Let's do n^2 * K comparison.
    
    # knn_indices shape (n, K)
    # We want to compute a matrix overlap_matrix (n, n) where entry (i, j) is the number of shared neighbors.
    
    # Method: For each i, j, count common elements in knn_indices[i] and knn_indices[j].
    # This is O(n^2 * K). With n=50-100, K=6, this is fast.
    
    # Vectorized approach using np.isin might be slow due to overhead.
    # Let's use a loop over n, which is cleaner and often faster for small n than full vectorization of set ops.
    
    overlap_matrix = np.zeros((n, n))
    if K > 0:
        for i in range(n):
            # Get unique neighbors of i
            neighbors_i = knn_indices[i]
            # Check which neighbors of j are in neighbors_i for all j
            # neighbors_i shape (K,)
            # knn_indices shape (n, K)
            # np.isin(knn_indices, neighbors_i) -> shape (n, K) bool
            # np.sum(axis=1) -> shape (n,) int
            
            # Note: np.isin is available in numpy >= 1.13
            is_neighbor = np.isin(knn_indices, neighbors_i)
            overlap_matrix[i, :] = np.sum(is_neighbor, axis=1)
            
    # Normalize overlap to [0, 1]
    # Max possible overlap is K (if they share all neighbors)
    if K > 0:
        overlap_norm = overlap_matrix / K
    else:
        overlap_norm = np.zeros((n, n))
        
    # Refine scaling factor: Inversely proportional to median shared neighbors
    # Calculate the median of the overlap counts (excluding diagonal self-overlap if desired, but diagonal is K)
    # We look at the distribution of overlaps to set the scale.
    if n > 1 and K > 0:
        # Flatten overlap matrix and exclude diagonal for a better estimate of inter-node cohesion
        flat_overlaps = overlap_matrix.flatten()
        # Mask out diagonal
        diag_indices = np.arange(n)
        diag_overlaps = overlap_matrix[diag_indices, diag_indices] # Should be K or close to it
        
        # Use median of off-diagonal elements
        off_diag_overlaps = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    off_diag_overlaps.append(overlap_matrix[i, j])
        
        if len(off_diag_overlaps) > 0:
            median_shared = np.median(off_diag_overlaps)
        else:
            median_shared = K / 2.0 # Fallback
            
        if median_shared > 0:
            # Base scale normalized by median. 
            # If median shared is high, the "default" cohesion is high, so we don't need as much exponential boost.
            # If median shared is low, differences are more significant, so we boost more.
            base_geom_scale = 3.0
            geom_cluster_scale = base_geom_scale / median_shared
        else:
            geom_cluster_scale = 1.5 # Fallback
    else:
        geom_cluster_scale = 1.5 # Fallback

    # Apply exponential bonus
    # Higher overlap -> higher score
    geom_cluster_bonus = np.exp(geom_cluster_scale * overlap_norm)
    
    # Apply bonus
    # Do not apply to depot edges
    geom_depot_mask = np.ones_like(geom_cluster_bonus)
    geom_depot_mask[:, 0] = 1.0
    geom_depot_mask[0, :] = 1.0
    
    heuristic_matrix *= geom_cluster_bonus * geom_depot_mask

    # 11. Demand-Symmetry Bonus
    # Boosts edges between nodes with similar demand-to-distance ratios relative to their local neighborhood.
    # This encourages forming routes that balance load distribution locally.
    # Scale by inverse of global demand coefficient of variation.
    
    # Calculate global demand coefficient of variation for customers (excluding depot)
    customer_demands = demands[1:]
    if len(customer_demands) > 1 and np.mean(customer_demands) > 0:
        demand_std = np.std(customer_demands)
        demand_mean = np.mean(customer_demands)
        cv_demand = demand_std / demand_mean
        # Inverse CV scale: if CV is low (uniform demands), scale is high (stronger bonus for symmetry)
        # if CV is high (varied demands), scale is low (weaker bonus)
        # Add small epsilon to avoid division by zero
        inv_cv_scale = 1.0 / (cv_demand + 1e-9)
    else:
        inv_cv_scale = 1.0 # Default if all demands are same or few nodes
        
    # Calculate local demand-to-distance ratio for each node
    # We define "distance" as the average distance to K nearest neighbors
    if K > 0:
        # Get distances to KNNs
        knn_distances = distance_matrix[np.arange(n)[:, np.newaxis], knn_indices] # Shape (n, K)
        avg_knn_dist = np.mean(knn_distances, axis=1) # Shape (n,)
        
        # Avoid division by zero
        avg_knn_dist_safe = avg_knn_dist + epsilon
        
        # Demand-to-average-neighbor-distance ratio
        local_ratio = demands / avg_knn_dist_safe # Shape (n,)
    else:
        # Fallback if K=0 (e.g., n=1)
        local_ratio = np.zeros(n)

    # Compute pairwise absolute difference in local ratios
    ratio_diff = np.abs(local_ratio[:, np.newaxis] - local_ratio[np.newaxis, :]) # Shape (n, n)
    
    # Normalize ratio diff by the max possible ratio diff or global mean to keep exp argument reasonable
    max_ratio_diff = np.max(ratio_diff)
    if max_ratio_diff > 0:
        norm_ratio_diff = ratio_diff / max_ratio_diff
    else:
        norm_ratio_diff = np.zeros((n, n))
        
    # Bonus: exp(-scale * norm_diff)
    # Scale factor scaled by inverse CV
    base_symmetry_scale = 2.0
    symmetry_scale = base_symmetry_scale * inv_cv_scale
    
    demand_symmetry_bonus = np.exp(-symmetry_scale * norm_ratio_diff)
    
    # Do not apply to depot edges
    symmetry_depot_mask = np.ones_like(demand_symmetry_bonus)
    symmetry_depot_mask[:, 0] = 1.0
    symmetry_depot_mask[0, :] = 1.0
    
    heuristic_matrix *= demand_symmetry_bonus * symmetry_depot_mask

    # 12. Cluster-Boundary Penalty
    # Penalizes edges that cross between distinct KNN-defined clusters.
    # Clusters are defined by the centroid of a node's KNNs.
    # If edge (i, j) connects nodes whose cluster centroids are far apart relative to the edge length,
    # it is likely a boundary-crossing edge.
    
    # Centroids are already computed in Step 2: cluster_centers (n, 2)
    
    # Compute pairwise distances between cluster centroids
    # Shape (n, n)
    # cluster_centers[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :]
    centroid_diff = cluster_centers[:, np.newaxis, :] - cluster_centers[np.newaxis, :, :] # Shape (n, n, 2)
    centroid_dist = np.sqrt(np.sum(centroid_diff ** 2, axis=2)) # Shape (n, n)
    
    # Normalize centroid distance by global scale
    # Use max_pairwise_dist computed earlier
    max_centroid_dist = np.max(centroid_dist[1:, 1:]) # Exclude depot from max calculation if possible
    if max_centroid_dist <= 0:
        max_centroid_dist = 1.0
        
    norm_centroid_dist = centroid_dist / max_centroid_dist
    
    # Define penalty strength
    # If centroid distance is large, nodes are in different clusters -> Penalty
    # If centroid distance is small, nodes are in same cluster -> No Penalty (or boost)
    
    # We want to penalize crossing. 
    # Let's use an exponential decay: exp(-k_cross * norm_centroid_dist)
    # This naturally favors short centroid distances (same cluster) and penalizes long ones.
    
    # Scale factor for penalty
    base_cross_scale = 2.0
    
    # Adaptive scale based on problem spread? 
    # If clusters are naturally spread out, we might want to penalize less?
    # Or more? Let's keep it simple and static for now.
    
    cluster_boundary_penalty = np.exp(-base_cross_scale * norm_centroid_dist)
    
    # Apply penalty only to non-depot edges
    boundary_depot_mask = np.ones_like(cluster_boundary_penalty)
    boundary_depot_mask[:, 0] = 1.0
    boundary_depot_mask[0, :] = 1.0
    
    heuristic_matrix *= cluster_boundary_penalty * boundary_depot_mask

    # Ensure diagonal is 0
    np.fill_diagonal(heuristic_matrix, 0)
    
    # Handle NaNs and negatives
    heuristic_matrix = np.nan_to_num(heuristic_matrix, nan=0.0, posinf=1e9, neginf=0.0)
    heuristic_matrix = np.maximum(heuristic_matrix, 1e-9)
    
    return heuristic_matrix

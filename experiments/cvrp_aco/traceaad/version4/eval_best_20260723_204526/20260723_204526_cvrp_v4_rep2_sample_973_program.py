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
    depot_coords = coordinates[0, :]
    
    # --- Part 1: Proximity/Nearest-Neighbor Sweep Component ---
    # Calculate the mean distance from each node i to all other nodes
    dist_sum = np.sum(distance_matrix, axis=1)  # (n,)
    mean_dist = dist_sum / (n - 1)  # (n,)
    
    # Avoid division by zero
    mean_dist = np.where(mean_dist == 0, 1e-9, mean_dist)
    
    # Create matrix of mean distances for broadcasting
    mean_dist_col = mean_dist[:, np.newaxis]  # (n, 1)
    
    # Proximity ratio: distance relative to mean
    dist_ratio = distance_matrix / mean_dist_col  # (n, n)
    
    # Proximity score: higher is better (shorter edge relative to average)
    # Using 1 / dist_ratio gives higher scores for shorter edges
    proximity_score = 1.0 / (dist_ratio + 1e-9)
    
    # --- Part 2: Demand-Density Component ---
    # Reward edges to nodes with high demand relative to their distance
    dist_safe = np.where(distance_matrix == 0, 1e-9, distance_matrix)  # (n, n)
    
    # Demand matrix: each row i has demands for all j
    demands_row = demands[np.newaxis, :]  # (1, n)
    
    # Demand density: demand[j] / dist[i, j]
    demand_density = demands_row / dist_safe  # (n, n)
    
    # Scale demand density by capacity to keep values in a reasonable range
    demand_density_scaled = demand_density / capacity
    
    # --- Part 3: K-Nearest-Neighbor Graph Bias (Vectorized) ---
    # Reward edges connecting each node to its spatially closest neighbors.
    # This encourages compact sub-tours and avoids long jumps across the graph.
    
    knn_bonus = np.ones((n, n), dtype=np.float64)
    
    if n >= 3:
        # Choose K neighbors. Adaptive to ensure we don't go out of bounds.
        # K=5 is a reasonable heuristic for local connectivity.
        k = min(5, n - 1)
        
        # Distance matrix copy to avoid modifying input or issues with diagonals
        d_mat = distance_matrix.copy()
        np.fill_diagonal(d_mat, np.inf) # Ignore self-loops for neighbor finding
        
        # np.argpartition(d_mat, k, axis=1) partitions each row such that the first k elements are the smallest.
        # Note: kth element is at index k, so elements at indices 0..k-1 are the k smallest.
        if k < n - 1:
            k_indices = np.argpartition(d_mat, k, axis=1)[:, :k]
        else:
            # If k >= n-1, all other nodes are neighbors
            k_indices = np.argsort(d_mat, axis=1)[:, :k]
        
        # Vectorized assignment of KNN bonus using advanced indexing
        # indices for rows: np.arange(n)[:, None] -> (n, 1)
        # indices for cols: k_indices -> (n, k)
        
        row_idx = np.arange(n)[:, np.newaxis]
        
        # Assign 2.0 to the KNN entries
        knn_bonus[row_idx, k_indices] = 2.0
            
    # --- Part 4: Capacity Feasibility Penalty ---
    # Penalize edges between two customers whose combined demand exceeds capacity.
    
    # Compute pairwise sum of demands
    demands_col = demands[:, np.newaxis]
    demand_sum = demands_col + demands_row  # (n, n)
    
    # Identify pairs where demand_sum > capacity
    # Note: Depot (node 0) has demand 0, so edges involving depot are rarely penalized
    capacity_penalty = np.where(demand_sum > capacity, 0.1, 1.0)
    
    # --- Part 5: Sector-Angle Bias Component ---
    # Reward edges that maintain a consistent radial direction from the depot.
    # This promotes sweep-like tour structures and reduces route crossing.
    
    # Vectors from depot to each node
    vectors_from_depot = coordinates - depot_coords[np.newaxis, :]  # (n, 2)
    
    # Radial angles of each node relative to depot
    radial_angles = np.arctan2(vectors_from_depot[:, 1], vectors_from_depot[:, 0])  # (n,)
    
    # Vectors for each edge i -> j
    edge_vectors = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]  # (n, n, 2)
    
    # Edge directions (angles)
    edge_angles = np.arctan2(edge_vectors[:, :, 1], edge_vectors[:, :, 0])  # (n, n)
    
    # Angle difference between the edge direction and the starting node's radial angle
    radial_angles_col = radial_angles[:, np.newaxis]  # (n, 1)
    angle_diff = np.abs(edge_angles - radial_angles_col)  # (n, n)
    
    # Normalize angle difference to [0, pi]
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    
    # Sector bias: reward small angle differences (edges continuing in radial direction)
    # Use a decay function: exp(-alpha * angle_diff)
    alpha_sector = 2.0  # Tuning parameter
    sector_bias = np.exp(-alpha_sector * angle_diff)
    
    # --- Part 6: Cluster Cohesion Component ---
    # Reward edges between nodes that share similar radial angles from the depot.
    # This reinforces the sector-angle bias by ensuring that nodes visited consecutively
    # are within the same angular sector, promoting contiguous cluster visits.
    
    # Compute pairwise radial angle differences
    # radial_angles shape: (n,)
    # radial_diff[i, j] = |radial_angles[i] - radial_angles[j]|
    radial_diff = np.abs(radial_angles[:, np.newaxis] - radial_angles[np.newaxis, :])
    
    # Normalize to [0, pi]
    radial_diff = np.minimum(radial_diff, 2 * np.pi - radial_diff)
    
    # Cluster cohesion: Gaussian decay based on radial angle difference
    # Beta controls the width of the angular sector considered "cohesive"
    beta = 1.0
    cluster_cohesion = np.exp(-beta * (radial_diff ** 2))
    
    # --- Part 7: Sweep-Sequence Bias ---
    # Penalize edges causing large increases in cumulative radial angle.
    # We encourage monotonic angular ordering.
    
    # Compute signed angular difference from i to j
    # We want to reward small positive steps (forward sweep) and penalize large steps or backward steps.
    
    # To handle the wrap-around at 2*pi, we compute the minimal signed difference
    # diff = angles[j] - angles[i]
    # Normalize to [-pi, pi]
    
    ang_diff_raw = radial_angles[np.newaxis, :] - radial_angles[:, np.newaxis] # (n, n)
    
    # Normalize to [-pi, pi]
    ang_diff_norm = np.remainder(ang_diff_raw + np.pi, 2 * np.pi) - np.pi
    
    # Sweep sequence bias: exp(-gamma * |angular_difference|)
    # Using a sharper gamma than cluster cohesion to enforce tighter sequencing.
    
    gamma_sweep = 2.0
    sweep_sequence_bias = np.exp(-gamma_sweep * np.abs(ang_diff_norm))
    
    # --- Part 8: Depot-Proximity Return Bias ---
    # Boost edges leading back to the depot (node 0) from nodes geographically close to the depot.
    # This facilitates efficient route closure.
    
    # Distance from each node i to the depot (node 0)
    dist_to_depot = distance_matrix[:, 0]  # (n,)
    
    # Normalize distance to depot for scaling
    max_dist_to_depot = np.max(dist_to_depot) if n > 1 else 1.0
    if max_dist_to_depot == 0:
        max_dist_to_depot = 1e-9
        
    # Create a bias factor that decays exponentially with distance to depot
    # alpha_depot controls the strength of the bias
    alpha_depot = 2.0
    depot_return_bias = np.exp(-alpha_depot * (dist_to_depot / max_dist_to_depot))
    
    # Apply this bias only to the column corresponding to the depot (index 0)
    # We create a matrix of ones and only modify the first column
    depot_return_matrix = np.ones((n, n), dtype=np.float64)
    depot_return_matrix[:, 0] = depot_return_bias
    
    # --- Part 9: Geometric Detour Penalty (Enhanced with Distance Weighting) ---
    # Penalize edges (i, j) if the angle formed by the depot-i-j triangle indicates
    # a significant deviation from a straight radial sweep.
    # Calculated using the cross-product of vectors from the depot.
    
    # Vectors from depot to all nodes: shape (n, 2)
    # vec_i corresponds to row i, vec_j corresponds to row j
    
    # We need vectors for i (shape n, 2) and vectors for j (shape n, 2)
    # To compute cross product for all pairs (i, j):
    # Let V be (n, 2). 
    # We want cross_prod(V[i], V[j]) for all i, j.
    # Cross product in 2D: x1*y2 - x2*y1
    
    # V[:, 0] is x-coordinates relative to depot (n,)
    # V[:, 1] is y-coordinates relative to depot (n,)
    
    x_coords = vectors_from_depot[:, 0]  # (n,)
    y_coords = vectors_from_depot[:, 1]  # (n,)
    
    # Compute cross products for all pairs (i, j)
    # cross[i, j] = x_i * y_j - x_j * y_i
    # Using outer products:
    # x_i * y_j -> outer(x, y)
    # x_j * y_i -> outer(y, x)
    
    cross_matrix = np.outer(x_coords, y_coords) - np.outer(y_coords, x_coords)  # (n, n)
    
    # The magnitude of the cross product is proportional to the area of the triangle.
    # To normalize, we divide by the product of the lengths of the two vectors (distances from depot).
    # dist_from_depot[i] * dist_from_depot[j]
    
    dist_from_depot = np.linalg.norm(vectors_from_depot, axis=1)  # (n,)
    
    # Avoid division by zero for depot itself (dist=0)
    dist_from_depot_safe = np.where(dist_from_depot == 0, 1e-9, dist_from_depot)
    
    # Normalized cross product magnitude
    # norm_cross[i, j] = |cross[i, j]| / (dist[i] * dist[j])
    # This value ranges from 0 (collinear with depot) to 1 (orthogonal to depot vectors)
    
    dist_outer = np.outer(dist_from_depot_safe, dist_from_depot_safe)  # (n, n)
    
    normalized_cross = np.abs(cross_matrix) / dist_outer
    
    # Clamp to 0-1 range just in case of numerical noise > 1
    normalized_cross = np.clip(normalized_cross, 0, 1)
    
    # Introduce a distance-weighted scaling factor:
    # The penalty is amplified for edges far from the depot and reduced for edges close to the depot.
    # scaling_factor[i, j] = (dist_from_depot[i] + dist_from_depot[j]) / (2 * max_dist_to_depot)
    # This factor is 0 for edges at the depot and 1 for edges at the max distance.
    
    # Create distance matrix for scaling
    dist_from_depot_col = dist_from_depot[:, np.newaxis]  # (n, 1)
    dist_from_depot_row = dist_from_depot[np.newaxis, :]  # (1, n)
    
    # Scaling factor: average normalized distance from depot
    scaling_factor = (dist_from_depot_col + dist_from_depot_row) / (2 * max_dist_to_depot)  # (n, n)
    
    # Apply penalty with distance weighting: exp(-alpha_detour * normalized_cross * scaling_factor)
    # A large normalized cross product AND large distance -> strong penalty.
    # A large normalized cross product BUT small distance -> weaker penalty.
    alpha_detour = 1.5
    geometric_detour_penalty = np.exp(-alpha_detour * normalized_cross * scaling_factor)
    
    # --- Part 10: Capacity-Aware Lookahead Heuristic (Enhanced) ---
    # Boost edges to nodes with demands low enough to likely allow for at least one 
    # subsequent customer visit. Penalizes edges that consume nearly all residual capacity.
    # Enhanced: Integrate dynamic residual capacity distribution estimate.
    
    # 1. Calculate residual capacity at node i.
    #    Approximate residual capacity as: Capacity - demand[i]. 
    residual_capacity = capacity - demands_col  # (n, 1)
    residual_capacity = np.maximum(residual_capacity, 0)
    
    # 2. Evaluate demand of target node j relative to residual capacity at i.
    #    Handle division by zero if residual_capacity is 0.
    res_cap_safe = np.where(residual_capacity == 0, 1e-9, residual_capacity) # (n, 1)
    
    demand_ratio = demands_row / res_cap_safe  # (n, n)
    
    # 3. Define the lookahead bias:
    #    Exponentially decay the bias as the ratio increases.
    alpha_lookahead = 5.0
    lookahead_bias = np.exp(-alpha_lookahead * demand_ratio)
    
    # 4. Dynamic Residual Capacity Distribution Estimate (New)
    #    Calculate the mean demand of all customers (excluding depot).
    customer_demands = demands[1:] if n > 1 else demands
    mean_customer_demand = np.mean(customer_demands) if len(customer_demands) > 0 else 0.0
    
    #    Create a mask for non-depot customer nodes to apply this heuristic logic.
    #    We want to penalize visiting a high-demand node (demand > mean) if we have 
    #    enough space to visit a typical node, encouraging us to pick lighter nodes first
    #    to pack more into the route.
    
    #    Identify edges to nodes with demand > mean_customer_demand
    is_high_demand = demands_row > mean_customer_demand # (n, n)
    
    #    Identify edges from nodes with significant residual capacity (e.g., > mean_customer_demand)
    #    If residual is small, we are forced to close or pick large nodes, so penalty is less relevant.
    has_significant_space = residual_capacity > mean_customer_demand # (n, 1)
    
    #    Penalty factor: If we have space for an average node, but choose a high-demand node,
    #    we penalize this move slightly to encourage picking lighter nodes that leave room for others.
    
    #    Extra demand over mean: (demand[j] - mean_demand)
    #    Penalty scale: exp(-lambda * (demand[j] - mean_demand) / capacity)
    #    This is only active if has_significant_space is true.
    
    extra_demand = np.maximum(demands_row - mean_customer_demand, 0) # (n, n)
    lambda_penalty = 2.0
    
    #    Compute penalty matrix
    penalty_matrix = np.exp(-lambda_penalty * (extra_demand / capacity))
    
    #    Apply penalty only where we had space to choose lightly
    #    Expand has_significant_space to (n, n)
    has_space_matrix = has_significant_space # Broadcasting (n, 1) to (n, n)
    
    #    Combine: If we have space, use the penalty_matrix. Otherwise, use 1.0 (no penalty).
    dynamic_capacity_penalty = np.where(has_space_matrix, penalty_matrix, 1.0)
    
    # 5. Additional Boost: "Feasible Continuation" Check.
    #    If the remaining capacity after visiting j is enough to visit at least the 
    #    "average" demand customer, give a boost.
    
    #    Remaining capacity after visiting j: residual_capacity[i] - demand[j]
    remaining_after_j = residual_capacity - demands_row # (n, n)
    
    #    Boost if remaining_after_j >= mean_customer_demand
    continuation_boost = np.ones((n, n), dtype=np.float64)
    
    #    Mask where remaining capacity is sufficient for at least one more average customer
    mask_sufficient = remaining_after_j >= mean_customer_demand
    continuation_boost[mask_sufficient] *= 1.5
    
    #    Combine lookahead components
    capacity_lookahead_heuristic = lookahead_bias * dynamic_capacity_penalty * continuation_boost
    
    # --- Part 11: Nearest-Neighbor Density Gradient (Enhanced with Demand-Weighted Sparsity) ---
    # Boosts edges connecting nodes in sparse regions to their closest available neighbors.
    # Enhanced: Prioritizes connections to high-demand nodes in sparse regions only if they fit within residual capacity.
    
    density_gradient_bias = np.ones((n, n), dtype=np.float64)
    
    if n >= 2:
        # Compute nearest neighbor distance for each node
        # Copy distance matrix and set diagonal to infinity
        d_mat_nn = distance_matrix.copy()
        np.fill_diagonal(d_mat_nn, np.inf)
        
        # Min distance to any other node
        min_dist = np.min(d_mat_nn, axis=1)  # (n,)
        
        # Indices of nearest neighbors
        nn_indices = np.argmin(d_mat_nn, axis=1)  # (n,)
        
        # Calculate density gradient boost
        # Boost factor: inversely proportional to distance, scaled by a factor
        
        max_min_dist = np.max(min_dist) if np.size(min_dist) > 0 else 1e-9
        if max_min_dist == 0:
            max_min_dist = 1e-9
            
        # Sparsity factor: 1.0 for dense nodes, higher for sparse nodes
        sparsity_factor = max_min_dist / (min_dist + 1e-9)
        
        # Demand-weighted sparsity metric:
        # Scale the sparsity boost by the demand of the target node relative to capacity.
        # This encourages connecting to high-demand nodes in sparse regions if capacity allows.
        target_demands = demands[nn_indices]  # (n,)
        demand_weight = target_demands / capacity  # (n,)
        
        # Capacity constraint: Only apply boost if target node fits in residual capacity
        # Residual capacity at source node i: capacity - demand[i]
        residual_cap_at_source = capacity - demands  # (n,)
        fits_capacity = target_demands <= residual_cap_at_source  # (n,)
        
        # Combine sparsity, demand weight, and capacity check
        # Base boost is (1.0 + sparsity_factor)
        # Enhanced boost is Base * (1.0 + demand_weight) if fits_capacity, else Base * 1.0
        base_boost = 1.0 + sparsity_factor
        enhanced_boost = np.where(fits_capacity, base_boost * (1.0 + demand_weight), base_boost)
        
        # Apply boost only to the nearest neighbor edge
        row_idx = np.arange(n)
        
        # Vectorized assignment
        density_gradient_bias[row_idx, nn_indices] *= enhanced_boost
        
    # --- Part 12: Depot-Transition Efficiency Bias (Refined) ---
    # Penalize returning to the depot if the current node's angular sector likely 
    # contains other unvisited customers and capacity allows.
    # Refinement: Only penalize if residual capacity > min demand in that sector.
    
    depot_transition_bias = np.ones((n, n), dtype=np.float64)
    
    if n >= 3:
        # 1. Estimate Angular Sector Density
        # For each node i, estimate how "crowded" its angular sector is.
        # We define a sector width. A simple heuristic is to look at the average angular 
        # gap between neighbors in the sorted angular list.
        
        # Sort angles to find local density
        sorted_indices = np.argsort(radial_angles)
        sorted_angles = radial_angles[sorted_indices]
        
        # Calculate angular gaps between consecutive sorted nodes
        # Wrap-around gap is needed for circular nature
        gaps = np.diff(sorted_angles)
        wrap_around_gap = 2 * np.pi - (sorted_angles[-1] - sorted_angles[0])
        all_gaps = np.concatenate([gaps, [wrap_around_gap]])
        
        # Average gap size represents typical sector size
        avg_gap = np.mean(all_gaps)
        
        # Define a threshold sector width, e.g., 2 * average gap
        sector_width = 2.0 * avg_gap
        
        # For each node i, count how many other nodes j fall within [angle_i - w/2, angle_i + w/2]
        # This is a static estimate based on instance geometry, assuming uniform unvisited probability.
        
        # Compute pairwise angular differences (already done as ang_diff_norm)
        # ang_diff_norm[i, j] is the signed difference from i to j in [-pi, pi]
        # We want to count j such that |ang_diff_norm[i, j]| < sector_width / 2
        
        half_width = sector_width / 2.0
        in_sector = np.abs(ang_diff_norm) < half_width # (n, n)
        
        # Count nodes in sector for each i (excluding self)
        in_sector_mask = in_sector.copy()
        np.fill_diagonal(in_sector_mask, 0)
        sector_counts = np.sum(in_sector_mask, axis=1) # (n,)
        
        # Normalize counts to a probability or density factor
        # If sector_counts[i] is high, returning to depot is "risky" (might miss customers)
        max_count = np.max(sector_counts) if np.any(sector_counts) else 1.0
        if max_count == 0:
            max_count = 1.0
            
        density_factor = sector_counts / max_count # (n,)
        
        # 2. Capacity Constraint Check (Refined)
        # Calculate residual capacity at node i: capacity - demand[i]
        residual_cap_i = capacity - demands # (n,)
        
        # For each node i, find the minimum demand of customers in its sector
        # We create a mask of nodes in the sector for each i
        # min_demand_in_sector[i] = min(demand[j] for j in sector of i)
        
        # Initialize with a large value
        min_demand_in_sector = np.full(n, np.inf)
        
        # Identify valid customer nodes (not depot)
        is_customer = np.arange(n) > 0
        
        # For each node i, look at its sector neighbors j
        # We want min(demands[j]) where in_sector[i, j] is true and j is a customer
        
        # Construct a matrix of demands, replacing non-sector nodes with inf
        demands_masked = demands.copy()
        
        # For each row i, set demands[j] to inf if j is not in sector or j is depot
        # in_sector is (n, n)
        # We want to ignore j=0 (depot) and j not in sector
        
        # Create a valid mask: customer AND in sector
        valid_sector_mask = in_sector & is_customer[np.newaxis, :] # (n, n)
        
        # Set demands to inf for invalid nodes
        demands_for_min = np.where(valid_sector_mask, demands, np.inf)
        
        # Compute min along axis 1
        # If no customers in sector, min_demand will be inf
        min_demand_in_sector = np.min(demands_for_min, axis=1)
        
        # Replace inf with capacity + 1 (so condition residual > min_demand is false)
        # This handles cases where no customers are in the sector
        min_demand_in_sector = np.where(np.isinf(min_demand_in_sector), capacity + 1, min_demand_in_sector)
        
        # Condition: Penalize only if residual_cap_i > min_demand_in_sector
        # This means we have enough capacity to visit at least the smallest customer in the sector
        can_visit_smallest = residual_cap_i > min_demand_in_sector # (n,)
        can_visit_smallest = can_visit_smallest.astype(np.float64)
        
        # 3. Combine into Bias Matrix
        # Apply penalty only to edges going to depot (column 0)
        # Penalty magnitude: exp(-alpha_dt * density_factor * can_visit_smallest)
        
        alpha_dt = 2.0
        penalty_values = np.exp(-alpha_dt * density_factor * can_visit_smallest)
        
        # Apply to column 0
        depot_transition_bias[:, 0] = penalty_values
        
        # Ensure other columns remain 1.0 (no bias applied)
        # depot_transition_bias is already initialized to ones.

    # --- Part 13: Dynamic Capacity-Margin Gradient ---
    # Refinement: Scales edge weights by the ratio of residual capacity after visiting node j 
    # to a dynamic percentile of feasible remaining demands.
    
    # 1. Identify all non-depot customers
    customer_indices = np.arange(1, n)
    customer_demands_arr = demands[customer_indices]
    
    # If there are no customers, fallback to small value
    if len(customer_demands_arr) == 0:
        median_demand = 1e-9
    else:
        # 2. Calculate residual capacity for each node i
        residual_cap_i = capacity - demands  # (n,)
        
        # 3. For each node i, find the percentile of feasible demands
        # Feasible demands are those <= residual_cap_i
        # We want the 25th percentile (lower bound of easy customers) to be more conservative.
        
        # Create a matrix of feasible demands for each source node i
        # If demand[j] > residual_cap_i[j], it's not feasible to visit j then continue?
        # Actually, this heuristic is about visiting j. The "margin" is for AFTER j.
        # So we look at residual AFTER visiting j: Cap - Dem[i] - Dem[j].
        # We want this residual to be >= some threshold.
        
        # Let's define the threshold based on the 25th percentile of all customer demands
        # that are potentially visitable (i.e., <= Capacity).
        
        feasible_demands = customer_demands_arr[customer_demands_arr <= capacity]
        if len(feasible_demands) == 0:
            # If no customer fits, just use max demand
            dynamic_threshold = np.max(customer_demands_arr)
        else:
            # Use 25th percentile as a "small" demand benchmark
            dynamic_threshold = np.percentile(feasible_demands, 25)
            if dynamic_threshold < 1e-9:
                dynamic_threshold = 1e-9

        # 4. Calculate Residual Capacity After Visiting j
        # residual_after[i, j] = capacity - demand[i] - demand[j]
        residual_after = capacity - demands_col - demands_row  # (n, n)
        residual_after = np.maximum(residual_after, 0)
        
        # 5. Calculate Margin Ratio
        # ratio[i, j] = residual_after[i, j] / dynamic_threshold
        margin_ratio = residual_after / dynamic_threshold
        
        # 6. Define Penalty/Boost based on Margin Ratio
        # We want to penalize if margin_ratio < 1.0 (cannot fit a "small" customer)
        # Increase sharpness (beta) to strongly penalize tight fits.
        
        beta_margin = 4.0 # Increased sharpness
        penalty_exponent = -beta_margin * np.maximum(0, 1.0 - margin_ratio)
        capacity_margin_bias = np.exp(penalty_exponent)
        
        # 7. Exception for Nearest Neighbor in Sparse Clusters
        # Relax the penalty if j is the nearest neighbor of i AND i is in a sparse region.
        
        # Re-use min_dist and nn_indices from Part 11 if available.
        # Recompute efficiently to keep logic self-contained in this block or reuse vars.
        # We will recompute to be safe and explicit.
        
        d_mat_nn_sparse = distance_matrix.copy()
        np.fill_diagonal(d_mat_nn_sparse, np.inf)
        min_dist_sparse = np.min(d_mat_nn_sparse, axis=1)
        nn_indices_sparse = np.argmin(d_mat_nn_sparse, axis=1)
        
        # Define "sparse region" threshold: e.g., top 20% of nearest neighbor distances
        if np.size(min_dist_sparse) > 0:
            sparse_threshold = np.percentile(min_dist_sparse, 75)
        else:
            sparse_threshold = 0.0
            
        # Mask: is_source_sparse[i] is True if min_dist[i] > sparse_threshold
        is_source_sparse = min_dist_sparse > sparse_threshold  # (n,)
        
        # Create a mask for edges (i, j) where j is NN of i and i is sparse
        # is_nn_edge[i, j] is True if j == nn_indices[i]
        is_nn_edge = np.zeros((n, n), dtype=bool)
        row_idx_nn = np.arange(n)
        is_nn_edge[row_idx_nn, nn_indices_sparse] = True
        
        # Combine conditions: Relax penalty if (is_sparse AND is_nn)
        relax_mask = is_source_sparse[:, np.newaxis] & is_nn_edge # (n, n)
        
        # If relax_mask is True, set capacity_margin_bias to 1.0 (neutral)
        # Otherwise, keep the calculated bias
        capacity_margin_bias = np.where(relax_mask, 1.0, capacity_margin_bias)
    
    # --- Part 14: Neighbor-Demand Correlation (Refined with Distance Weighting and Dynamic Sigma) ---
    # Penalize edges to nodes with demands significantly deviating from the mean demand 
    # of the current node's spatial K-nearest neighbors.
    # Refinement: Weight the mean demand calculation by the inverse of distance, creating a 
    # distance-weighted local demand profile that more accurately reflects immediate geographic 
    # packing constraints and reduces the penalty impact of distant neighbors in sparse clusters.
    # Additional Refinement: Scale the penalty strength (sigma) dynamically based on local spatial density.
    
    demand_correlation_bias = np.ones((n, n), dtype=np.float64)
    
    if n >= 3:
        # Use the same K and k_indices as in Part 3 for consistency
        k = min(5, n - 1)
        d_mat = distance_matrix.copy()
        np.fill_diagonal(d_mat, np.inf)
        
        if k < n - 1:
            k_indices_corr = np.argpartition(d_mat, k, axis=1)[:, :k]
        else:
            k_indices_corr = np.argsort(d_mat, axis=1)[:, :k]
            
        # Get distances to K-nearest neighbors for each node i: shape (n, k)
        # Use advanced indexing to extract distances from d_mat
        row_indices = np.arange(n)[:, np.newaxis] # (n, 1)
        knn_distances = d_mat[row_indices, k_indices_corr] # (n, k)
        
        # Calculate inverse distance weights for normalization
        # Avoid division by zero
        weights = 1.0 / (knn_distances + 1e-9) # (n, k)
        
        # Get demands of K-nearest neighbors for each node i: shape (n, k)
        demands_knn = demands[k_indices_corr] # (n, k)
        
        # Compute weighted mean demand for each node i
        # weighted_mean = sum(weights * demands) / sum(weights)
        weighted_demands = weights * demands_knn # (n, k)
        sum_weights = np.sum(weights, axis=1, keepdims=True) # (n, 1)
        
        # Avoid division by zero in sum_weights (unlikely but safe)
        sum_weights = np.where(sum_weights == 0, 1e-9, sum_weights)
        
        weighted_mean_demand = np.sum(weighted_demands, axis=1, keepdims=True) / sum_weights # (n, 1)
        
        # Compute absolute deviation of target node j's demand from the source node i's weighted neighbor mean
        deviation = np.abs(demands_row - weighted_mean_demand) # (n, n)
        
        # Normalize deviation by capacity to scale appropriately
        normalized_deviation = deviation / (capacity + 1e-9)
        
        # Dynamic Sigma Calculation based on Local Density
        # Calculate average distance to KNNs for each node i
        avg_knn_dist = np.mean(knn_distances, axis=1) # (n,)
        
        # Normalize avg_knn_dist to [0, 1] range relative to instance size
        max_avg_dist = np.max(avg_knn_dist) if np.max(avg_knn_dist) > 0 else 1e-9
        normalized_density = avg_knn_dist / max_avg_dist # Higher value -> More Sparse
        
        # Define base sigma and scaling factor
        base_sigma = 2.0
        density_scaling_factor = 1.5 # Max increase in sigma for dense regions
        
        # Sigma decreases as normalized_density increases (i.e., as region becomes sparser)
        # sigma = base_sigma / (1 + density_scaling_factor * normalized_density)
        # This ensures sigma is high (strict) for dense regions (low dist) and low (lenient) for sparse regions (high dist)
        dynamic_sigma = base_sigma / (1.0 + density_scaling_factor * normalized_density)[:, np.newaxis] # (n, 1)
        
        # Apply Gaussian penalty with dynamic sigma
        demand_correlation_bias = np.exp(-dynamic_sigma * (normalized_deviation ** 2))
        
    # --- Combine All Components ---
    # Multiply all heuristic components
    heuristic_matrix = (proximity_score * demand_density_scaled * knn_bonus * 
                        capacity_penalty * sector_bias * cluster_cohesion * sweep_sequence_bias * 
                        depot_return_matrix * geometric_detour_penalty * capacity_lookahead_heuristic * 
                        density_gradient_bias * depot_transition_bias * capacity_margin_bias *
                        demand_correlation_bias)
    
    # Ensure non-negative values
    heuristic_matrix = np.maximum(heuristic_matrix, 0)
    
    # Set diagonal and self-loops to 0 or very low to avoid sampling
    np.fill_diagonal(heuristic_matrix, 0)
    
    # Replace zeros/negatives with a small positive value for sampling stability
    heuristic_matrix = np.where(heuristic_matrix <= 0, 1e-9, heuristic_matrix)
    
    return heuristic_matrix

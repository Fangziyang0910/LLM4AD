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
    eps = 1e-10
    
    # --- Part 1: Inverse Distance Base ---
    # Shorter edges are preferred.
    dist_safe = np.where(distance_matrix == 0, eps, distance_matrix)
    dist_inv = 1.0 / dist_safe
    
    # --- Part 2: Progress Centrality Component ---
    # Estimate how "central" a node is relative to unvisited customers.
    # Moving to a node that is closer to the cluster of remaining nodes is favorable.
    mean_dist = np.mean(distance_matrix, axis=1)
    
    # Avoid division by zero
    mean_dist_row = mean_dist[np.newaxis, :] + eps
    mean_dist_col = mean_dist[:, np.newaxis]
    
    # Progress factor: mean_dist[i] / mean_dist[j]
    # If mean_dist[j] < mean_dist[i], factor > 1 (good).
    progress_matrix = mean_dist_col / mean_dist_row
    
    # Logarithmic scaling to prevent extreme values
    log_progress = np.log1p(progress_matrix)
    
    # --- Part 3: Angular Consistency Component ---
    # Favor edges that continue in the same direction relative to the depot.
    angular_bonus = np.ones((n, n), dtype=np.float64)
    
    if n > 1:
        depot_coords = coordinates[0]
        
        # Direction from depot to each node i
        depot_to_i = coordinates - depot_coords
        
        # Normalize vectors for angle computation
        norm_depot_to_i = np.linalg.norm(depot_to_i, axis=1, keepdims=True)
        norm_depot_to_i = np.where(norm_depot_to_i == 0, eps, norm_depot_to_i)
        depot_to_i_norm = depot_to_i / norm_depot_to_i
        
        # For each i and j, compute direction i->j
        i_to_j = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]
        
        norm_i_to_j = np.linalg.norm(i_to_j, axis=2, keepdims=True)
        norm_i_to_j = np.where(norm_i_to_j == 0, eps, norm_i_to_j)
        i_to_j_norm = i_to_j / norm_i_to_j
        
        # Dot product to get cosine of angle between vector (Depot->i) and (i->j)
        dot_product = np.sum(depot_to_i_norm[:, np.newaxis, :] * i_to_j_norm, axis=2)
        
        # Angular bonus: reward high dot product (small angle / straight path)
        angular_bonus_val = 0.5 + 0.5 * np.clip(dot_product, -1, 1)
        
        # Apply angular bonus only for i > 0 (departing from customer)
        mask_i_non_depot = (np.arange(n)[:, np.newaxis] > 0)
        
        angular_bonus = np.where(mask_i_non_depot, angular_bonus_val, 1.0)
    
    # --- Part 4: Hybrid Capacity and Spatial Kernel ---
    
    # Precompute normalized demands for source (i) and destination (j)
    demand_i_norm = demands / (capacity + eps)
    demand_j_norm = demands / (capacity + eps)
    
    # 4a. Adaptive Spatial Kernel for Customer-to-Customer edges
    # High-demand source nodes have tighter spatial attraction (smaller sigma).
    # High-demand destination nodes are penalized to preserve residual capacity.
    spatial_kernel = np.ones((n, n))
    
    if n > 1 and np.max(demands[1:]) > 0:
        # Determine k for nearest neighbors
        k = max(2, int(capacity / np.max(demands[1:])))
        actual_k = min(k, n - 1)
        
        if actual_k > 0:
            # Calculate local scale parameter sigma_i for each node
            # sigma_i is the MEDIAN distance to its 'actual_k' nearest neighbors
            
            # Sort distances for each row to find k-nearest neighbors
            dist_sorted = np.sort(distance_matrix, axis=1)
            
            # Exclude self-distance (index 0) and take next 'actual_k' distances
            end_idx = min(actual_k + 1, n)
            k_nearest_dists = dist_sorted[:, 1:end_idx]
            
            # Compute MEDIAN distance (sigma_base) for each node
            sigma_base = np.median(k_nearest_dists, axis=1)
            
            # Avoid division by zero or extremely small sigma
            sigma_base = np.maximum(sigma_base, 1e-6)
            
            # Adapt sigma based on SOURCE and DESTINATION demand using geometric mean approach.
            # sigma_ij = sigma_base_i * exp(-max(demand_i_norm, demand_j_norm))
            # This ensures edges involving high-demand nodes (either source or dest) have tighter constraints.
            max_demand_norm = np.maximum(demand_i_norm[:, np.newaxis], demand_j_norm[np.newaxis, :])
            adaptive_factor = np.exp(-max_demand_norm)
            
            adaptive_sigma = sigma_base[:, np.newaxis] * adaptive_factor
            
            # Ensure stability
            adaptive_sigma = np.maximum(adaptive_sigma, 1e-6)
            
            # Gaussian Kernel: exp(-d^2 / (2 * sigma^2))
            sigma_sq = adaptive_sigma ** 2
            
            # Compute exponent: -0.5 * (distance_matrix ** 2) / sigma_sq
            exponent = -0.5 * (distance_matrix ** 2) / (sigma_sq + eps)
            
            # Compute Gaussian weights
            gaussian_weights = np.exp(exponent)
            
            # Apply destination demand penalty for customer-to-customer edges
            # Penalty: 1 / (1 + demand_j_norm)
            # This makes high-demand destinations less attractive to preserve capacity.
            demand_penalty = 1.0 / (1.0 + demand_j_norm[np.newaxis, :])
            
            # Combine Gaussian attraction with destination penalty
            spatial_kernel = gaussian_weights * demand_penalty

    # 4b. Depot Boost for Customer-to-Depot edges
    # Encourage returning to depot when source node has high demand (high residual cost).
    # Applied when j == 0 (destination is depot).
    # Formula: exp(demand_i_norm)
    depot_boost = np.exp(demand_i_norm)[:, np.newaxis] # Shape (n, 1)
    
    # Create mask for destination index j=0 (Depot)
    # j=0 corresponds to the first column of the matrix
    mask_j_depot = np.zeros((n, n), dtype=bool)
    mask_j_depot[:, 0] = True
    
    # Combine: Use Depot Boost for j=0, Spatial Kernel (with penalty) for j>0
    combined_kernel = np.where(mask_j_depot, depot_boost, spatial_kernel)
    
    # --- Combine all factors ---
    # Heuristic = Base_Dist * Progress * Angular * Combined_Capacity_Spatial_Kernel
    heuristic_matrix = dist_inv * (1.0 + 0.5 * log_progress) * angular_bonus * combined_kernel
    
    # Set self-loops in the main matrix to 0
    np.fill_diagonal(heuristic_matrix, 0)
    
    # Ensure no negative values and minimum threshold for ACO compatibility
    heuristic_matrix = np.maximum(heuristic_matrix, 1e-9)
    
    # Handle any potential inf/nan
    heuristic_matrix = np.where(np.isfinite(heuristic_matrix), heuristic_matrix, 0)
    heuristic_matrix = np.maximum(heuristic_matrix, 1e-9)
    
    return heuristic_matrix

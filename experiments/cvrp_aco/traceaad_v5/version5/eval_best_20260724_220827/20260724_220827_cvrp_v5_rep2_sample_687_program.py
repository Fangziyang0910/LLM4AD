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
    
    # Handle trivial cases
    if n <= 1:
        return np.ones((n, n))
        
    # Depot is node 0
    depot_idx = 0
    
    # Initialize heuristic matrix
    heur = np.zeros((n, n), dtype=np.float64)
    
    # --- Component 1-5: Fused Geometric and Capacity Potential ---
    
    depot_coord = coordinates[0]
    diffs = coordinates - depot_coord
    
    # 1. Angular Potential (Sweeping)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    delta_angle = angles[:, None] - angles[None, :]
    
    # Normalize angular difference to [-pi, pi]
    sin_diff = np.sin(delta_angle)
    cos_diff = np.cos(delta_angle)
    delta_angle_wrapped = np.arctan2(sin_diff, cos_diff)
    
    # Angular Similarity Bonus: exp(-|dtheta|)
    angular_bonus = np.exp(-np.abs(delta_angle_wrapped))
    
    # 2. Radial Potential
    radii = np.linalg.norm(diffs, axis=1)
    delta_radius = radii[:, None] - radii[None, :]
    
    # Compute Median Absolute Deviation (MAD) of radii for robust scaling
    median_radii = np.median(radii)
    mad_radii = np.median(np.abs(radii - median_radii))
    if mad_radii < 1e-6:
        mad_radii = 1.0
        
    # Radial Coherence Bonus: exp(-|dr|/MAD)
    radial_bonus = np.exp(-np.abs(delta_radius) / mad_radii)
    
    # 3. Directional Alignment Bonus (from Primary Synthesis)
    # Calculate angle of edge vector (j - i)
    edge_diffs = coordinates[None, :, :] - coordinates[:, None, :] # Shape (n, n, 2)
    edge_angles = np.arctan2(edge_diffs[:, :, 1], edge_diffs[:, :, 0])
    
    # Calculate angle of radial vector at destination node j
    # rad_angles[j] is the angle of vector (coordinates[j] - depot)
    rad_angles = angles[None, :] # Shape (1, n) -> broadcasts to (n, n)
    
    # Delta between edge direction and destination radial direction
    delta_angle_radial = edge_angles - rad_angles
    
    # Normalize to [-pi, pi]
    sin_diff_rad = np.sin(delta_angle_radial)
    cos_diff_rad = np.cos(delta_angle_radial)
    delta_angle_radial_wrapped = np.arctan2(sin_diff_rad, cos_diff_rad)
    
    # Bonus for alignment with radial direction (straight out/in from depot)
    directional_bonus = np.exp(-np.abs(delta_angle_radial_wrapped))
    
    # 4. Distance & Angular Decoupled Terms
    epsilon = 1e-6
    
    # 5. Capacity Penalty
    pair_demand = demands[:, None] + demands[None, :]
    normalized_demand = pair_demand / capacity
    capacity_factor = np.maximum(1.0 - (normalized_demand ** 3), 0.0)
    
    # Fused Calculation:
    # heur = (1 / dist) * (1 / |dtheta_sweep|) * angular_bonus * radial_bonus * directional_bonus * capacity_factor
    heur = (1.0 / (distance_matrix + epsilon)) * \
           (1.0 / (np.abs(delta_angle_wrapped) + epsilon)) * \
           angular_bonus * \
           radial_bonus * \
           directional_bonus * \
           capacity_factor
    
    # --- Component 6: 2D Radial-Angular Inter-Cluster Transition Penalty ---
    # Instead of pure angular sectors, use a 2D grid of angle and log-radius.
    # This groups nodes by both direction and distance, creating more physical clusters.
    
    n_sectors = 24
    n_radial_bins = 8
    
    # Normalize angles to [0, 2pi) for binning
    norm_angles = np.mod(angles, 2 * np.pi)
    
    # Assign angular sector index
    node_sectors = (norm_angles / (2 * np.pi) * n_sectors).astype(int)
    node_sectors = np.mod(node_sectors, n_sectors)
    
    # Assign radial bin index using logarithmic scaling for better distribution
    # Avoid log(0) by adding epsilon and clamping min radius to epsilon
    log_radii = np.log(radii + epsilon)
    min_log_r = np.min(log_radii)
    max_log_r = np.max(log_radii)
    
    # Avoid division by zero if all nodes are at same distance
    log_range = max_log_r - min_log_r
    if log_range < 1e-6:
        log_range = 1.0
        
    # Map log-radii to [0, n_radial_bins)
    node_radial_bins = ((log_radii - min_log_r) / log_range * n_radial_bins).astype(int)
    node_radial_bins = np.clip(node_radial_bins, 0, n_radial_bins - 1)
    
    # Create sector difference matrix (angular)
    sector_diff = np.abs(node_sectors[:, None] - node_sectors[None, :])
    # Handle wrap-around for circular sectors
    sector_diff = np.minimum(sector_diff, n_sectors - sector_diff)
    
    # Create radial bin difference matrix
    radial_bin_diff = np.abs(node_radial_bins[:, None] - node_radial_bins[None, :])
    
    # Combine angular and radial differences into a single cluster distance metric
    # Use Weighted Manhattan Distance: angular_diff + radial_diff (with implicit weight 1 for each normalized unit)
    # This decouples the penalties compared to Euclidean.
    cluster_dist = sector_diff + radial_bin_diff
    
    # Penalty factor: exp(-beta * cluster_dist)
    # Reduce beta to 1.0 to prevent over-penalizing necessary radial transitions in dense clusters.
    beta = 1.0
    inter_cluster_penalty = np.exp(-beta * cluster_dist)

    # Apply penalty to the heuristic directly
    heur *= inter_cluster_penalty

    # --- Component 6.5: Radial Boundary Penalty ---
    # Penalize edges connecting nodes in significantly different radial bins (diff > 2).
    # This discourages zig-zagging between inner and outer rings.
    radial_boundary_penalty_threshold = 2.0
    radial_boundary_penalty_strength = 0.5  # Exponential decay factor
    
    # Identify edges crossing multiple radial bins
    radial_crossing_mask = radial_bin_diff > radial_boundary_penalty_threshold
    # Calculate penalty for crossing edges: exp(-strength * (diff - threshold))
    # For non-crossing edges, penalty is 1.0 (no change)
    radial_penalty = np.ones_like(radial_bin_diff)
    radial_penalty[radial_crossing_mask] = np.exp(-radial_boundary_penalty_strength * (radial_bin_diff[radial_crossing_mask] - radial_boundary_penalty_threshold))
    
    # Apply radial boundary penalty
    heur *= radial_penalty

    # --- Component 7: Consolidated Boost Application (Fused In-Place with Global Median Decay) ---
    
    # Identify feasible edges
    feasible_mask = (pair_demand <= capacity)
    
    # To find the nearest feasible neighbor, we mask out infeasible distances with inf
    masked_dist = np.where(feasible_mask, distance_matrix, np.inf)
    
    # Ensure self-loops are not chosen as neighbors
    np.fill_diagonal(masked_dist, np.inf)
    
    # Find argmin
    nearest_feasible_neighbors = np.argmin(masked_dist, axis=1)
    
    # Check if any feasible neighbor exists
    min_dists = np.min(masked_dist, axis=1)
    has_feasible = min_dists < np.inf
    
    # Initialize neighbor_counts and global_median_dist for Component 8 usage
    neighbor_counts = np.zeros(n)
    global_median_dist = 1.0

    # --- Nearest Feasible Neighbor (NFN) Boost ---
    if np.any(has_feasible):
        i_indices = np.where(has_feasible)[0]
        j_indices = nearest_feasible_neighbors[i_indices]
        
        # Get the distances to the nearest feasible neighbors
        dists_to_nearest = masked_dist[i_indices, j_indices]
        
        # Compute median of min feasible distances for robust instance-scale normalization
        # Median is more robust to outliers in sparse CVRP instances than mean distance
        valid_min_dists = min_dists[has_feasible]
        if len(valid_min_dists) == 0:
            global_median_dist = 1.0
        else:
            global_median_dist = np.median(valid_min_dists)
            
        # Distance-adaptive boost with median normalization: 
        # boost = 1.0 + global_median_dist * (1 / (1 + dist))
        boost = 1.0 + global_median_dist * (1.0 / (1.0 + dists_to_nearest))
        
        # Apply NFN boost directly to heur matrix (In-place)
        heur[i_indices, j_indices] *= boost
        
        # --- Cluster Continuity Bonus ---
        # For each node i, define a local cluster radius as 1.5 * dist(i, nearest_feasible_neighbor(i))
        # If node j is within this radius of i (and feasible), apply a continuity boost.
        
        alpha = 0.5  # Boost factor
        
        # Cluster radii for nodes that have feasible neighbors
        cluster_radii = np.full(n, np.inf)
        cluster_radii[has_feasible] = 1.5 * min_dists[has_feasible]
        
        # Create a mask where distance[i,j] <= cluster_radius[i]
        # This implies j is "close" to i relative to i's nearest neighbor
        proximity_mask = (distance_matrix <= cluster_radii[:, None])
        
        # Only consider feasible edges for cluster continuity
        cluster_continuity_mask = proximity_mask & feasible_mask
        
        # --- Synthesized Boost: Geometric Decay (Global Median) & Density Normalization ---
        
        # 1. Use global median distance for geometric scaling of the exponential decay.
        # This enforces robustness against local sparsity outliers as per reference rollback.
        if global_median_dist < 1e-6:
            global_median_dist = 1.0
            
        # Exponential decay based on geometric proximity scaled by global median
        geo_decay = np.exp(-distance_matrix / global_median_dist)
            
        # 2. Use local neighbor count for topological density modulation.
        # Count feasible neighbors within the cluster radius for each node.
        neighbor_counts = np.sum(cluster_continuity_mask, axis=1)
        
        # Determine max count for normalization across all nodes
        max_count = np.max(neighbor_counts)
        if max_count < 1e-6:
            max_count = 1.0
            
        # Calculate density factor based on local node density
        density_factor = neighbor_counts / max_count
        
        # Combine geometric decay and density factor
        # Boost is stronger for shorter edges (geo_decay) and in denser areas (density_factor)
        combined_boost = 1.0 + alpha * geo_decay * density_factor[:, None]
            
        # Apply combined boost directly to heur matrix where mask is True (In-place boolean indexing)
        heur[cluster_continuity_mask] *= combined_boost[cluster_continuity_mask]

    # --- Component 8: Return-to-Depot Boost (Synthesized from Reference) ---
    
    # Identify sparse nodes: nodes with low neighbor counts in the cluster continuity mask.
    # These nodes are isolated and benefit from a higher probability of returning to the depot.
    sparse_mask = (neighbor_counts < 2)
    
    # Calculate distance to depot for each node (column 0 of distance_matrix)
    dist_to_depot = distance_matrix[:, 0]
    
    # Compute boost factor for return-to-depot edges for sparse nodes.
    # boost = 1 + gamma * (dist_to_depot / median_dist)
    # This makes the return option more attractive for distant, sparse nodes.
    gamma = 1.0
    
    # Avoid division by zero if global_median_dist is too small
    safe_median = global_median_dist if global_median_dist > 1e-6 else 1.0
    
    # Apply boost to column 0 (edges to depot) for sparse nodes using vectorized indexing
    sparse_indices = np.where(sparse_mask)[0]
    if len(sparse_indices) > 0:
        heur[sparse_indices, 0] *= (1.0 + gamma * (dist_to_depot[sparse_indices] / safe_median))

    # --- Cleaning ---
    # Ensure no NaN or Inf
    heur = np.nan_to_num(heur, nan=0.0, posinf=1e10, neginf=0.0)
    
    # Ensure non-negative
    heur = np.maximum(heur, 0.0)
    
    # Zero out diagonal (self-loops)
    np.fill_diagonal(heur, 0.0)
    
    return heur

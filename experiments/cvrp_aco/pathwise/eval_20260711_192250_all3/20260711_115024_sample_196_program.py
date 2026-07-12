import random
import math
import scipy
try:
    import torch
except Exception:
    torch = None
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
    
    # 1. Base Distance Term: Sharp decay for local clustering (-1.5 exponent)
    safe_dist = distance_matrix + 1e-9
    dist_factor = np.power(safe_dist, -1.5)
    
    # 2. Node Weights (Polynomial Urgency)
    # Formula: 1 / (1 - util)^1.5
    node_weight = np.ones(n)
    eps = 1e-5
    
    if n > 1:
        cust_idx = np.arange(1, n)
        d_cust = demands[cust_idx]
        
        # Utilization ratio
        util = d_cust / capacity
        
        # Clip to avoid division by zero and infinity
        util_clipped = np.clip(util, 0.0, 1.0 - eps)
        
        # Polynomial Urgency component
        node_weight[cust_idx] = np.power(1.0 - util_clipped, -1.5)

    # 3. Balanced Demand Factor for Internal Edges (Linear Scaling)
    # Factor = Capacity / demand. Linear scaling for destination bias.
    demand_safe = demands.copy()
    demand_safe[0] = capacity
    balanced_demand_factor = capacity / np.maximum(demand_safe, 1e-8)
    
    # Matrix for destination scaling
    balanced_demand_matrix = np.outer(np.ones(n), balanced_demand_factor)

    # 4. Construct Heuristic Matrix
    heur = np.zeros_like(distance_matrix)
    cust_idx = np.arange(1, n)

    # --- Customer-to-Customer Edges (i, j) where i, j > 0 ---
    # Directive: Use joint tightness exponent 2.7 (Increased Stability/Pressure)
    tightness_exp = 2.7
    
    if n > 2:
        # Demand structures for pair-wise checks
        demand_row = demands[cust_idx].reshape(-1, 1)
        demand_col = demands[cust_idx].reshape(1, -1)
        demand_sum = demand_row + demand_col
        
        # Strict Infeasibility Masking
        exceeds_capacity = demand_sum > capacity
        compat_sub = np.where(exceeds_capacity, 1e-9, 1.0)
        
        # Joint Tightness Term: Exponent 2.7
        utilization_pair = demand_sum / capacity
        utilization_pair_clipped = np.clip(utilization_pair, 0.0, 1.0 - eps)
        base_tightness = 1.0 / (1.0 - utilization_pair_clipped)
        tightness_factor = np.power(base_tightness, tightness_exp)
        
        # Destination Node Urgency Bias
        dest_urgency_matrix = np.outer(np.ones(n-1), node_weight[cust_idx])
        
        # Distance submatrix
        dist_sub = dist_factor[np.ix_(cust_idx, cust_idx)]
        
        # Balanced Demand Factor for C2C (Linear scaling)
        balanced_sub = balanced_demand_matrix[np.ix_(cust_idx, cust_idx)]
        
        # Spatial Cosine Similarity Boost (from rollout_10_0_1_0)
        cust_coords = coordinates[cust_idx, :] # (n-1, 2)
        norms = np.linalg.norm(cust_coords, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1e-9, norms)
        unit_vectors = cust_coords / norms
        
        # Cosine similarity matrix (n-1, n-1)
        cos_sim = unit_vectors @ unit_vectors.T
        
        # Boost: 1 + cos_sim (ranges 0 to 2)
        spatial_boost = 1.0 + cos_sim
        
        # NEW: Depot-Relative Angle Bias with Increased Strength (k_angle=3.0)
        # Promotes edges that continue the radial sweep from the depot, 
        # helping to form coherent routes rather than just clusters.
        depot_coord = coordinates[0:1, :] # (1, 2)
        vecs_from_depot = cust_coords - depot_coord # (n-1, 2)
        norms_depot = np.linalg.norm(vecs_from_depot, axis=1, keepdims=True)
        norms_depot = np.where(norms_depot < 1e-9, 1e-9, norms_depot)
        unit_from_depot = vecs_from_depot / norms_depot
        
        # Cosine similarity of angles relative to depot
        angle_sim = unit_from_depot @ unit_from_depot.T
        
        # Boost: 1.0 + 3.0 * angle_sim (Scaled up for tighter angular continuity)
        k_angle = 3.0
        angle_boost = 1.0 + k_angle * angle_sim
        
        # Combine: Dist * Dest_Urgency * Spatial_Bias * Angle_Bias * Joint_Tightness * Balanced_Demand * Compatibility
        heur[np.ix_(cust_idx, cust_idx)] = (dist_sub * 
                                             dest_urgency_matrix * 
                                             spatial_boost * 
                                             angle_boost * 
                                             tightness_factor * 
                                             balanced_sub * 
                                             compat_sub)

    # --- Depot-to-Customer Edges (0, j) where j > 0 ---
    # Retain sqrt(capacity/demand) scaling for robust initiation
    depot_demand_exp = 0.5
    
    if n > 1:
        dep_to_cust_dist = dist_factor[0, cust_idx]
        dep_to_cust_compat = np.where(demands[cust_idx] <= capacity, 1.0, 1e-9)
        
        # Demand Factor: (capacity/demand)^0.5
        cust_demands = demands[cust_idx]
        demand_factor_cust = np.power(capacity / np.maximum(cust_demands, 1e-8), depot_demand_exp)
        dep_to_cust_urgency = node_weight[cust_idx]
        
        # No spatial boost for depot edges
        heur[0, cust_idx] = dep_to_cust_dist * dep_to_cust_urgency * demand_factor_cust * dep_to_cust_compat
        
        # --- Customer-to-Depot Edges (i, 0) where i > 0 ---
        cust_to_dep_dist = dist_factor[cust_idx, 0]
        cust_to_dep_compat = np.where(demands[cust_idx] <= capacity, 1.0, 1e-9)
        
        # Source Urgency bias for closing route
        heur[cust_idx, 0] = cust_to_dep_dist * node_weight[cust_idx] * cust_to_dep_compat

    # 5. Post-processing
    np.fill_diagonal(heur, 0.0)
    heur[heur <= 0] = 1e-9
    
    return heur

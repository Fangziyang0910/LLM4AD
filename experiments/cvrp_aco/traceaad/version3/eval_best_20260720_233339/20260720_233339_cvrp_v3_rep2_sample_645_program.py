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
    
    # Ensure demands is 1D
    demands_flat = demands.flatten()
    
    # 1. Base Inverse Distance Component
    safe_dist = np.maximum(distance_matrix, 1e-9)
    inv_dist = 1.0 / safe_dist
    
    # 2. Demand Efficiency Component
    demand_j = demands_flat[np.newaxis, :]
    safe_demand_j = np.maximum(demand_j, 1e-9)
    demand_efficiency = safe_demand_j / safe_dist
    
    # 3. Demand Cluster Alignment Component
    
    demand_i = demands_flat[:, np.newaxis]
    pair_demand = demand_i + demand_j
    
    # Feasibility mask: Edge (i, j) is only valid if they can fit in a vehicle 
    feasible_mask = (pair_demand <= capacity)
    
    connectivity_factor = np.ones((n, n))
    
    if capacity > 0 and n > 1:
        # Get customer demands (excluding depot)
        cust_demands = demands_flat[1:]
        
        if len(cust_demands) > 0:
            # Calculate statistics of customer demands
            median_demand = np.median(cust_demands)
            
            # Calculate residual capacity for all edges
            # Residual is what's left in the vehicle after adding i and j
            # However, since we don't know the previous load, we treat this as 
            # "if i and j are the last two nodes" or "if i and j are the first two nodes".
            # A more robust interpretation for a static heuristic:
            # We want the residual capacity (Capacity - demand_i - demand_j) to be 
            # comparable to the typical demand of a remaining customer, suggesting
            # the route can be easily closed or continued by typical customers.
            
            residual_capacity = capacity - pair_demand # Can be negative if infeasible
            
            # We only care about feasible edges for this score
            # For feasible edges, residual is >= 0
            
            # Alignment Score:
            # Boost if residual is close to median demand (meaning we can likely fit one more typical customer)
            # Or if residual is very small (close to 0), meaning the route is full.
            # Or if residual is close to capacity (start of route).
            
            # Let's focus on the "typical fit" aspect.
            # Distance to median
            dist_to_median = np.abs(residual_capacity - median_demand)
            
            # Normalize distance. Max possible residual is capacity.
            # We want a high score when dist_to_median is small.
            # Using a Gaussian-like decay or inverse distance
            scale_factor = max(capacity, 1e-9)
            alignment_score = np.exp(-dist_to_median / scale_factor)
            
            # Additionally, reward small residuals (full route)
            # If residual < median_demand, it might be hard to fit another typical customer.
            # But it's good if it's close to 0.
            # Let's blend: High score if residual ~ 0 OR residual ~ median.
            
            dist_to_zero = np.abs(residual_capacity)
            score_zero = np.exp(-dist_to_zero / (scale_factor * 0.1)) # Sharp peak at 0
            
            # Combined alignment
            # Weighted average of fitting a typical customer vs finishing the route
            combined_align = 0.6 * alignment_score + 0.4 * score_zero
            
            # Normalize combined_align to range [0.5, 1.5] or similar for modulation
            # Max value is ~1.0 + 0.4 = 1.4? No, exp(0)=1.
            # So range is roughly [0.4, 1.0]
            # Let's scale it to be a multiplier around 1.0
            connectivity_factor = 1.0 + combined_align
            
            # Apply feasibility mask
            connectivity_factor[~feasible_mask] = 1e-6
            
        else:
            connectivity_factor = np.ones((n, n))
            connectivity_factor[~feasible_mask] = 1e-6
    else:
        feasible_mask = np.ones((n, n), dtype=bool)
        connectivity_factor[~feasible_mask] = 1e-6
        
    # 4. Geometric Components (Angular)
    
    # Angular Consistency relative to depot
    angular_factor = np.ones((n, n))
    
    if n > 0:
        depot_coords = coordinates[0]
        vectors_from_depot = coordinates - depot_coords
        norms = np.linalg.norm(vectors_from_depot, axis=1, keepdims=True)
        norms_safe = np.maximum(norms, 1e-9)
        normalized_vectors = vectors_from_depot / norms_safe
        cosine_similarity = np.dot(normalized_vectors, normalized_vectors.T)
        cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
        
        dist_from_depot = norms_safe.flatten()
        avg_dist_from_depot = (dist_from_depot[:, np.newaxis] + dist_from_depot[np.newaxis, :]) / 2.0
        max_dist = np.max(dist_from_depot) if np.max(dist_from_depot) > 0 else 1.0
        
        weight_factor = 1.0 + 2.0 * (avg_dist_from_depot / max_dist)
        k_base = 2.0
        k_weighted = k_base * weight_factor
        
        angular_factor = np.exp(k_weighted * cosine_similarity)
        
    # Neutralize geometric effects for depot edges
    depot_mask = np.zeros((n, n), dtype=bool)
    if n > 0:
        depot_mask[0, :] = True
        depot_mask[:, 0] = True
        
    angular_factor[depot_mask] = 1.0
    
    # 5. Smoothness Reward Component
    
    smoothness_factor = np.ones((n, n))
    
    if n > 1:
        depot_coords = coordinates[0]
        vec_depot_to_i = coordinates - depot_coords[np.newaxis, :]
        norm_incoming = np.linalg.norm(vec_depot_to_i, axis=1, keepdims=True)
        norm_incoming_safe = np.maximum(norm_incoming, 1e-9)
        unit_incoming = vec_depot_to_i / norm_incoming_safe 
        
        vec_ij = coordinates[np.newaxis, :, :] - coordinates[:, np.newaxis, :]
        norm_outgoing = np.linalg.norm(vec_ij, axis=2, keepdims=True)
        norm_outgoing_safe = np.maximum(norm_outgoing, 1e-9)
        unit_outgoing = vec_ij / norm_outgoing_safe 
        
        turn_cosine = np.sum(unit_incoming[:, np.newaxis, :] * unit_outgoing, axis=2)
        turn_cosine = np.clip(turn_cosine, -1.0, 1.0)
        
        k_smooth = 1.0
        avg_dist = np.mean(safe_dist[safe_dist > 1e-9]) if np.any(safe_dist > 1e-9) else 1.0
        inv_dist_norm = inv_dist * avg_dist 
        
        alpha = 2.0
        smoothness_term = (turn_cosine + 1.0) / 2.0
        
        safe_demand_dest = np.maximum(demands_flat, 1e-9)[np.newaxis, :]
        mean_demand = np.mean(demands_flat[1:]) if n > 1 else 1.0
        mean_demand = np.maximum(mean_demand, 1e-9)
        norm_demand_weight = mean_demand / safe_demand_dest
        
        smoothness_factor = 1.0 + alpha * inv_dist_norm * smoothness_term * norm_demand_weight
        
        smoothness_factor[depot_mask] = 1.0
        
    # 6. Combine Components
    base_score = demand_efficiency * connectivity_factor
    
    heur = base_score * angular_factor * inv_dist * smoothness_factor

    # 7. Special Handling for Depot Edges (0 -> j)
    if n > 1:
        cust_demands = demands_flat[1:]
        if capacity > 0:
            demand_ratio = cust_demands / capacity
            depot_edge_boost = 1.0 + 2.0 * demand_ratio
            heur[0, 1:] *= depot_edge_boost

    # 8. Final Cleanup
    if n > 0:
        np.fill_diagonal(heur, 0.0)
    
    heur = np.maximum(heur, 1e-9)
    
    heur = np.where(np.isfinite(heur), heur, 1e-9)
    
    return heur

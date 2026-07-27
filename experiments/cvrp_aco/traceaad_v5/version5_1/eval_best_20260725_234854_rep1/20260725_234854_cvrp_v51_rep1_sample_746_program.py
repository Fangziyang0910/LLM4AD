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
    if n <= 1:
        return distance_matrix.copy()

    # 1. Identify natural clusters using a simple demand-weighted k-means-like approach
    customer_coords = coordinates[1:]  # shape (n-1, 2)
    customer_demands = demands[1:]     # shape (n-1,)
    
    num_customers = n - 1
    if num_customers == 0:
        return np.zeros((n, n))

    # Determine number of clusters: roughly sqrt of customers or capped
    num_clusters = min(5, max(2, int(np.sqrt(num_customers))))
    
    # Initialize cluster centers using the first 'num_clusters' customers sorted by demand (descending)
    sorted_cust_indices = np.argsort(-customer_demands)[:num_clusters]
    centers = customer_coords[sorted_cust_indices]
    
    # Simple K-means iteration (limited steps for speed)
    for _ in range(3):
        # Compute distances from all customers to each center
        dist_to_centers = np.linalg.norm(customer_coords[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        # Assign customers to nearest center
        labels = np.argmin(dist_to_centers, axis=1)
        
        # Update centers
        new_centers = np.zeros_like(centers)
        valid = True
        for k in range(num_clusters):
            mask = (labels == k)
            if np.any(mask):
                new_centers[k] = customer_coords[mask].mean(axis=0)
            else:
                valid = False
                break
        
        centers = new_centers
        if not valid:
            break

    # 2. Compute "Cluster Proximity" Score for each edge
    # Assign each customer to its closest final cluster center
    dist_to_centers_final = np.linalg.norm(customer_coords[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    customer_cluster_assignments = np.argmin(dist_to_centers_final, axis=1) # shape (num_customers,)
    
    # Compute distance between cluster centers
    center_dist_matrix = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2) # shape (k, k)
    
    # Create a matrix of cluster distances for all customer pairs
    cluster_dist = center_dist_matrix[customer_cluster_assignments[:, np.newaxis], customer_cluster_assignments[np.newaxis, :]]
    
    # Inverse of cluster distance to favor intra-cluster edges
    cluster_urgency = 1.0 / (cluster_dist + 1e-6)
    
    # 3. Compute Angular Bias for Depot-Adjacent Edges
    depot_coord = coordinates[0]
    # Vectors from depot to all nodes
    v_depot_to_nodes = coordinates - depot_coord # shape (n, 2)
    
    # Precompute norms
    norm_depot_to_i = np.linalg.norm(v_depot_to_nodes, axis=1) # shape (n,)
    # Avoid division by zero for depot itself
    norm_depot_to_i[0] = 1.0
    
    # Compute dot products for all pairs (i, j)
    D = v_depot_to_nodes @ coordinates.T
    S_i = np.sum(v_depot_to_nodes * coordinates, axis=1) # Shape (n,)
    
    # Dot product for edge (i, j) is D[i, j] - S_i
    dot_prod = D - S_i[:, np.newaxis]
    
    epsilon = 1e-9
    # Denominator: norm_depot_to_i[i] * distance_matrix[i, j]
    denom = norm_depot_to_i[:, np.newaxis] * distance_matrix
    # Avoid division by zero
    denom = np.where(denom > epsilon, denom, epsilon)
    
    cosine_matrix = dot_prod / denom
    cosine_matrix = np.clip(cosine_matrix, -1.0, 1.0)
    
    alpha = 0.5
    angular_bias = 1.0 + alpha * cosine_matrix
    
    # 4. Combine Heuristics
    # Base heuristic: inverse distance
    # Use epsilon addition for numerical stability as requested
    heur = 1.0 / (distance_matrix + 1e-9)
    
    # Apply cluster urgency for customer-to-customer edges
    heur[1:, 1:] *= cluster_urgency
    
    # Apply Angular Bias to customer-to-customer edges
    heur[1:, 1:] *= angular_bias[1:, 1:]
    
    # Apply dampened non-linear capacity penalty to customer-to-customer edges ONLY
    # Penalty: np.exp(-0.1 * (demands[i] + demands[j]) / capacity)
    if num_customers > 0:
        # customer_demands shape (n-1,)
        # We need a matrix (n-1, n-1)
        sym_demand_sum = customer_demands[:, np.newaxis] + customer_demands[np.newaxis, :]
        dest_penalty = np.exp(-0.1 * sym_demand_sum / capacity)
        heur[1:, 1:] *= dest_penalty

    # Introduce demand-aware incentive for customer-to-depot edges
    # Enhanced with capacity-utilization-aware exponential factor
    # (1.0 + np.exp(-0.2 * customer_demands / capacity))
    depot_incentive = 1.0 + np.exp(-0.2 * customer_demands / capacity)
    heur[1:, 0] *= depot_incentive

    # Zero out self-loops explicitly as requested to prevent ants from visiting the current node
    np.fill_diagonal(heur, 0)
    
    # Normalize or ensure positivity
    heur = np.maximum(heur, 0)
    
    return heur

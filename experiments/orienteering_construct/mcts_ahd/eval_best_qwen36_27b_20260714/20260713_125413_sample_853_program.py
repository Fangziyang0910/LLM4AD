
import numpy as np
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray, prizes: np.ndarray, remaining_budget: float) -> int:
    """
    Design a novel constructive heuristic for the Orienteering Problem.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the route destination node.
    unvisited_nodes: Array of feasible unvisited node IDs. Visiting one of these nodes still leaves enough budget to return to the destination.
    distance_matrix: Pairwise Euclidean distance matrix of all nodes.
    prizes: Prize values of all nodes. The depot prize is 0.
    remaining_budget: Remaining travel budget before selecting the next node.

    Return:
    ID of the next node to visit.
    """
    if len(unvisited_nodes) == 0:
        return destination_node

    best_score = -np.inf
    best_node = destination_node

    eps = 1e-9

    # Pre-calculate distances from current to all unvisited
    dist_curr_to_unvisited = distance_matrix[current_node, unvisited_nodes]
    # Pre-calculate distances from all unvisited to destination
    dist_unvisited_to_dest = distance_matrix[destination_node, unvisited_nodes]

    # Calculate total cost for direct return (current -> node -> dest)
    total_costs_direct = dist_curr_to_unvisited + dist_unvisited_to_dest

    # Feasibility mask: total cost must be within remaining budget
    feasible_mask = total_costs_direct <= remaining_budget
    feasible_indices = np.where(feasible_mask)[0]

    if len(feasible_indices) == 0:
        return destination_node

    feasible_nodes = unvisited_nodes[feasible_indices]
    feasible_dist_curr = dist_curr_to_unvisited[feasible_indices]
    feasible_dist_dest = dist_unvisited_to_dest[feasible_indices]
    feasible_total_costs = total_costs_direct[feasible_indices]
    feasible_prizes = prizes[feasible_nodes]

    # Pre-fetch all unvisited nodes for lookahead
    all_unvisited = unvisited_nodes

    # Global normalization factor for distances
    max_dist = np.max(distance_matrix) + eps

    # Calculate Lookahead Potentials for all feasible candidates
    # We will compute two new metrics:
    # 1. Reachability Entropy: Measures the spread of reachable nodes from the candidate.
    # 2. Cluster Value Density: Sum of prizes of reachable nodes normalized by their average distance.

    reachability_entropy = np.zeros(len(feasible_nodes))
    cluster_value_density = np.zeros(len(feasible_nodes))

    for i, node_idx in enumerate(feasible_nodes):
        # Remaining budget after traveling current -> node_idx
        budget_after_visit = remaining_budget - feasible_dist_curr[i]

        if budget_after_visit <= eps:
            reachability_entropy[i] = 0.0
            cluster_value_density[i] = 0.0
            continue

        # Find other unvisited nodes (excluding the current candidate node_idx)
        other_mask = all_unvisited != node_idx
        if not np.any(other_mask):
            reachability_entropy[i] = 0.0
            cluster_value_density[i] = 0.0
            continue

        other_nodes = all_unvisited[other_mask]

        # Distances from current candidate to all other nodes
        dist_to_others = distance_matrix[node_idx, other_nodes]
        # Distances from other nodes to destination
        dist_others_to_dest = distance_matrix[destination_node, other_nodes]

        # Total cost to visit other node j and return to destination
        # Path: current -> node_idx -> other_node -> destination
        # Cost constraint for lookahead: dist(node_idx, other_node) + dist(other_node, dest) <= budget_after_visit

        cost_others = dist_to_others + dist_others_to_dest

        # Feasibility mask for other nodes
        feasible_other_mask = cost_others <= budget_after_visit

        if not np.any(feasible_other_mask):
            reachability_entropy[i] = 0.0
            cluster_value_density[i] = 0.0
            continue

        feasible_other_nodes = other_nodes[feasible_other_mask]
        feasible_other_prizes = prizes[feasible_other_nodes]
        feasible_other_dists = dist_to_others[feasible_other_mask]

        # --- Reachability Entropy Calculation ---
        # We bin the distances of reachable nodes into 3 buckets: Close, Medium, Far
        # to estimate the diversity of spatial reachability.
        if len(feasible_other_nodes) > 0:
            # Normalize distances for binning
            max_other_dist = np.max(feasible_other_dists) + eps
            normalized_dists = feasible_other_dists / max_other_dist

            # Bins: [0, 1/3), [1/3, 2/3), [2/3, 1]
            bin_edges = np.array([0, 1/3, 2/3, 1])
            # Assign bins
            bins = np.digitize(normalized_dists, bin_edges) - 1 # bins will be 0, 1, 2

            # Calculate probability distribution over bins
            counts, _ = np.histogram(bins, bins=3, range=(0, 2))
            total_count = np.sum(counts)
            probabilities = counts / (total_count + eps)

            # Shannon Entropy
            # Entropy = - sum(p * log(p))
            # To avoid log(0), we can clip probabilities
            probs_clipped = np.clip(probabilities, eps, 1.0)
            entropy = -np.sum(probs_clipped * np.log2(probs_clipped))

            # Normalize entropy by max possible entropy for 3 bins (log2(3))
            reachability_entropy[i] = entropy / np.log2(3)
        else:
            reachability_entropy[i] = 0.0

        # --- Cluster Value Density Calculation ---
        if len(feasible_other_nodes) > 0:
            # Sum of prizes
            total_cluster_prize = np.sum(feasible_other_prizes)
            # Average distance to these nodes
            avg_dist = np.mean(feasible_other_dists) + eps
            cluster_value_density[i] = total_cluster_prize / avg_dist
        else:
            cluster_value_density[i] = 0.0

    # Normalize potentials
    max_entropy = np.max(reachability_entropy) if len(reachability_entropy) > 0 else 1.0
    max_entropy = max(max_entropy, eps)
    normalized_entropy = reachability_entropy / max_entropy

    max_density = np.max(cluster_value_density) if len(cluster_value_density) > 0 else 1.0
    max_density = max(max_density, eps)
    normalized_density = cluster_value_density / max_density

    # --- Novel Dynamic Scaling Factor Calculation ---
    # Using a hyperbolic tangent based scaling for smoother transition
    # As budget shrinks, penalty sensitivity increases non-linearly
    reference_budget = max_dist * 2.0
    ratio = remaining_budget / (reference_budget + eps)
    # Map ratio [0, 1] to a sensitivity factor
    # When ratio is small (low budget), sensitivity should be high
    dynamic_sensitivity = 1.0 / (1.0 + np.exp(-10 * (ratio - 0.5)))

    # Base penalty coefficient
    base_penalty_coeff = 3.0
    penalty_coeff = base_penalty_coeff * dynamic_sensitivity

    # Base score components
    # 1. Immediate Efficiency: Prize / Distance from current
    immediate_efficiency = (feasible_prizes / (feasible_dist_curr + eps)) / max_dist

    # 2. Lookahead Components
    entropy_weight = 0.4
    density_weight = 0.6

    lookahead_score = entropy_weight * normalized_entropy + density_weight * normalized_density

    # Combine base score
    base_score = immediate_efficiency + lookahead_score

    # Penalty for budget consumption
    # Use hyperbolic penalty: 1 / (1 + k * remaining_budget) type logic or power law
    # Here we use a modified power law with dynamic coefficient
    consumption_ratio = feasible_total_costs / (remaining_budget + eps)
    # Ensure ratio doesn't explode if budget is near 0
    consumption_ratio = np.clip(consumption_ratio, 0, 1.0)

    # Penalty increases as consumption_ratio approaches 1
    penalty = penalty_coeff * (consumption_ratio ** 2.5)

    # Final Score
    final_scores = base_score - penalty

    # Select best node
    best_idx = np.argmax(final_scores)
    best_node = feasible_nodes[best_idx]

    return best_node

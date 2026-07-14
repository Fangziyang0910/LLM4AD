
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
    import numpy as np

    if len(unvisited_nodes) == 0:
        return destination_node

    # Filter nodes with positive prizes
    valid_nodes = unvisited_nodes[prizes[unvisited_nodes] > 0]

    if len(valid_nodes) == 0:
        return destination_node

    # Pre-calculate distances for valid nodes from current location
    dist_current_to_node = distance_matrix[current_node, valid_nodes]
    # Pre-calculate distances for valid nodes to destination
    dist_node_to_dest = distance_matrix[destination_node, valid_nodes]

    # Total cost to visit node and return to destination
    total_cost = dist_current_to_node + dist_node_to_dest

    # Filter feasible nodes based on budget
    feasible_mask = total_cost <= remaining_budget
    feasible_nodes = valid_nodes[feasible_mask]

    if len(feasible_nodes) == 0:
        return destination_node

    if len(feasible_nodes) == 1:
        return feasible_nodes[0]

    # --- Novel Hybrid Heuristic: Regret-Augmented Forward Potential ---

    n_feasible = len(feasible_nodes)
    feasible_prizes = prizes[feasible_nodes]
    visit_return_cost = total_cost[feasible_mask]

    # Sub-distance matrix for feasible nodes
    sub_dist_matrix = distance_matrix[feasible_nodes][:, feasible_nodes]

    # 1. Calculate "Regret-Augmented Forward Potential" using Look-Ahead Simulation
    forward_potentials = np.zeros(n_feasible)
    decay = 0.94  # Balanced decay

    for i in range(n_feasible):
        # Start simulation from node i
        current_sim_node_idx = i

        # Distance from current_node to node i
        start_dist = dist_current_to_node[feasible_mask][i]
        current_sim_dist_from_current = start_dist

        total_sim_prize = feasible_prizes[i]

        visited_sim = np.zeros(n_feasible, dtype=bool)
        visited_sim[i] = True

        # Track collected prizes for regret calculation
        collected_indices = [i]

        while True:
            unvisited_indices = np.where(~visited_sim)[0]
            if len(unvisited_indices) == 0:
                break

            # Distances from current sim node to unvisited feasible nodes
            dists = sub_dist_matrix[current_sim_node_idx, unvisited_indices]

            next_node_global_indices = feasible_nodes[unvisited_indices]
            dist_next_to_dest = distance_matrix[destination_node, next_node_global_indices]

            # Check feasibility:
            # Path: current -> ... -> current_sim_node -> next_node -> dest
            # Cost = current_sim_dist_from_current + dist_to_next + dist_next_to_dest
            costs_to_return = current_sim_dist_from_current + dists + dist_next_to_dest
            feasible_mask_sim = costs_to_return <= remaining_budget

            feasible_sim_indices = unvisited_indices[feasible_mask_sim]

            if len(feasible_sim_indices) == 0:
                break

            sim_prizes = feasible_prizes[feasible_sim_indices]
            sim_dists = dists[feasible_mask_sim]

            safe_sim_dists = np.maximum(sim_dists, 1e-9)

            # Selection Criterion for Simulation:
            # Efficiency * (1 + Cluster Bonus)
            # Using cubic distance term for cluster bonus
            efficiency_sim = sim_prizes / safe_sim_dists
            cluster_bonus = 1.0 / (1.0 + sim_dists ** 3)

            # Adjusted weight for cluster bonus
            select_score = efficiency_sim * (1.0 + 0.5 * cluster_bonus)

            best_sim_local_idx = np.argmax(select_score)
            next_node_local_idx = feasible_sim_indices[best_sim_local_idx]

            dist_to_next = sim_dists[best_sim_local_idx]
            current_sim_dist_from_current += dist_to_next

            # Accumulate prize with decay
            total_sim_prize += feasible_prizes[next_node_local_idx] * (decay ** dist_to_next)

            visited_sim[next_node_local_idx] = True
            collected_indices.append(next_node_local_idx)
            current_sim_node_idx = next_node_local_idx

        # 2. Calculate Regret Penalty for this branch
        # Regret is high if high-value nodes are left unvisited and are close to the visited cluster
        unvisited_mask = ~visited_sim
        unvisited_prizes = feasible_prizes[unvisited_mask]

        regret_penalty = 0.0
        if np.any(unvisited_mask):
            # Find max prize among unvisited
            max_missed_prize = np.max(unvisited_prizes)
            missed_indices = np.where(unvisited_mask)[0]

            # Calculate minimum distance from any visited node to any missed node
            # This represents how "close" we were to a high-value node we missed
            if len(collected_indices) > 0 and len(missed_indices) > 0:
                # Map local indices to global sub-matrix indices
                visited_local = collected_indices
                dist_visited_to_missed = sub_dist_matrix[np.ix_(visited_local, missed_indices)]
                min_dist_to_missed = np.min(dist_visited_to_missed)

                # Regret: High prize missed + Small distance missed = High Penalty
                # Normalize distance to avoid scale issues
                regret_penalty = max_missed_prize / (1.0 + min_dist_to_missed)

        # Forward Potential = Accumulated Prize - Regret Penalty
        forward_potentials[i] = total_sim_prize - 0.6 * regret_penalty

    # 3. Normalize Potential
    max_potential = np.max(forward_potentials)
    min_potential = np.min(forward_potentials)

    if max_potential > min_potential:
        normalized_potential = (forward_potentials - min_potential) / (max_potential - min_potential)
    else:
        normalized_potential = np.zeros(n_feasible)

    gamma = 0.75 # Weight for potential

    # 4. Calculate Effective Cost with "Bridge Bonus" (Inspired by No.1)
    # A node is a "bridge" if it is centrally located relative to other high-prize unvisited nodes.

    K = min(5, n_feasible - 1)

    # Create a mask to exclude self for each row
    mask_self = ~np.eye(n_feasible, dtype=bool)
    dist_matrix_no_self = np.where(mask_self, sub_dist_matrix, np.inf)

    # Weighted Distance = Sum(prize_j * dist_ij) / Sum(prize_j)
    # This measures centrality weighted by neighbor importance

    sum_prizes = np.sum(feasible_prizes)
    if sum_prizes > 0:
        # Weighted average distance to all other nodes, weighted by their prizes
        weighted_dist = np.sum(dist_matrix_no_self * feasible_prizes.reshape(1, -1), axis=1) / sum_prizes
    else:
        # Fallback to unweighted average distance
        weighted_dist = np.sum(dist_matrix_no_self, axis=1) / (n_feasible - 1)

    # Bridge Bonus: Inverse of weighted distance. Closer to high-prize centers = higher bonus.
    # Normalize the bonus to [0, 1]
    max_weighted_dist = np.max(weighted_dist)
    min_weighted_dist = np.min(weighted_dist)

    if max_weighted_dist > min_weighted_dist:
        bridge_bonus = 1.0 - ((weighted_dist - min_weighted_dist) / (max_weighted_dist - min_weighted_dist))
    else:
        bridge_bonus = 0.5 * np.ones(n_feasible)

    # Effective Cost = Base Cost * (1 - Bridge_Bonus * reduction_factor)
    # Bridge nodes are effectively cheaper
    reduction_factor = 0.2
    effective_cost = visit_return_cost * (1.0 - reduction_factor * bridge_bonus)

    # 5. Dynamic Budget Elasticity (Inspired by No.1)
    # Sigmoid function for budget sensitivity
    # Calculate budget ratio relative to the total cost of feasible options
    # Or simpler: relative to remaining budget vs a typical cost
    # Let's use a sigmoid that becomes conservative when remaining budget is low relative to the min cost

    min_cost = np.min(visit_return_cost)
    budget_ratio = remaining_budget / (min_cost + 1e-9)

    # Sigmoid: sharp drop in tolerance for cost when budget is tight
    # If budget_ratio > 2, elasticity is high. If < 1, elasticity drops.
    sigma = 1.0 / (1.0 + np.exp(-3 * (budget_ratio - 1.5)))

    # Scale sigma to range [0.3, 1.0]
    budget_elasticity = 0.3 + 0.7 * sigma

    # 6. Final Score Calculation
    # Score = (Prize * (1 + Gamma * Normalized_Potential) * Budget_Elasticity) / Effective_Cost

    safe_effective_cost = np.maximum(effective_cost, 1e-9)

    scores = (feasible_prizes * (1.0 + gamma * normalized_potential) * budget_elasticity) / safe_effective_cost

    best_idx = np.argmax(scores)

    return feasible_nodes[best_idx]

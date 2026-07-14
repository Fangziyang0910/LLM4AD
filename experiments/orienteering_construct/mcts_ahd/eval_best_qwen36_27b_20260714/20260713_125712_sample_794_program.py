
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

    best_node = destination_node
    best_score = -float('inf')

    unvisited_list = list(unvisited_nodes)
    n_unvisited = len(unvisited_list)

    # Pre-calculate destination distances for all unvisited nodes to save computation in loops
    dest_dists = distance_matrix[:, destination_node]

    # New Constants for the new scoring formula
    # Alpha for power-law decay of immediate distance
    alpha_dist = 2.5
    # K for sigmoid budget factor inflection point
    k_sigmoid = 0.5
    # Steepness for sigmoid budget factor
    steepness_sigmoid = 5.0

    for candidate in unvisited_list:
        dist_to_candidate = distance_matrix[current_node, candidate]
        dist_candidate_to_dest = dest_dists[candidate]

        # Feasibility check: Can we go Current -> Candidate -> Dest?
        if dist_to_candidate + dist_candidate_to_dest > remaining_budget:
            continue

        prize_candidate = prizes[candidate]
        budget_after_first_step = remaining_budget - dist_to_candidate

        # --- Strategy 1: Local Greediness (Prize / Distance^2) ---
        # Heavily penalizes distance in the lookahead to favor very close nodes
        sim_node_1 = candidate
        sim_budget_1 = budget_after_first_step
        total_prize_1 = prize_candidate
        visited_sim_1 = {candidate}

        for _ in range(n_unvisited):
            best_next_1 = -1
            max_eff_1 = -1.0

            for next_n in unvisited_list:
                if next_n in visited_sim_1:
                    continue

                d_to_next = distance_matrix[sim_node_1, next_n]
                d_next_to_dest = dest_dists[next_n]

                if d_to_next + d_next_to_dest > sim_budget_1:
                    continue

                # Ratio of prize to squared immediate step distance (stronger distance penalty)
                if d_to_next > 1e-9:
                    eff = prizes[next_n] / (d_to_next ** 2)
                else:
                    eff = float('inf')

                if eff > max_eff_1:
                    max_eff_1 = eff
                    best_next_1 = next_n

            if best_next_1 == -1:
                break

            visited_sim_1.add(best_next_1)
            total_prize_1 += prizes[best_next_1]
            sim_budget_1 -= distance_matrix[sim_node_1, best_next_1]
            sim_node_1 = best_next_1

        # --- Strategy 2: Detour Efficiency (Prize / Round-Trip) ---
        # Considers the cost of leaving the "direct" path to dest and returning
        sim_node_2 = candidate
        sim_budget_2 = budget_after_first_step
        total_prize_2 = prize_candidate
        visited_sim_2 = {candidate}

        for _ in range(n_unvisited):
            best_next_2 = -1
            max_val_2 = -1.0

            for next_n in unvisited_list:
                if next_n in visited_sim_2:
                    continue

                d_to_next = distance_matrix[sim_node_2, next_n]
                d_next_to_dest = dest_dists[next_n]

                if d_to_next + d_next_to_dest > sim_budget_2:
                    continue

                # Efficiency based on total "detour" cost relative to prize
                round_trip_cost = d_to_next + d_next_to_dest
                if round_trip_cost > 1e-9:
                    val = prizes[next_n] / round_trip_cost
                else:
                    val = float('inf')

                if val > max_val_2:
                    max_val_2 = val
                    best_next_2 = next_n

            if best_next_2 == -1:
                break

            visited_sim_2.add(best_next_2)
            total_prize_2 += prizes[best_next_2]
            sim_budget_2 -= distance_matrix[sim_node_2, best_next_2]
            sim_node_2 = best_next_2

        # --- Strategy 3: Budget Conservation (Prize * Log(Remaining Budget + 1)) ---
        # Uses logarithmic scaling for budget to diminish returns on extra budget
        sim_node_3 = candidate
        sim_budget_3 = budget_after_first_step
        total_prize_3 = prize_candidate
        visited_sim_3 = {candidate}

        for _ in range(n_unvisited):
            best_next_3 = -1
            max_val_3 = -1.0

            for next_n in unvisited_list:
                if next_n in visited_sim_3:
                    continue

                d_to_next = distance_matrix[sim_node_3, next_n]
                d_next_to_dest = dest_dists[next_n]

                if d_to_next + d_next_to_dest > sim_budget_3:
                    continue

                # Value is prize weighted by log of budget remaining after visiting this node
                budget_after_next = sim_budget_3 - d_to_next
                val = prizes[next_n] * np.log(1.0 + budget_after_next)

                if val > max_val_3:
                    max_val_3 = val
                    best_next_3 = next_n

            if best_next_3 == -1:
                break

            visited_sim_3.add(best_next_3)
            total_prize_3 += prizes[best_next_3]
            sim_budget_3 -= distance_matrix[sim_node_3, best_next_3]
            sim_node_3 = best_next_3

        # --- Hybrid Scoring ---
        # Take the maximum estimated prize from the three strategies
        total_path_value = max(total_prize_1, total_prize_2, total_prize_3)

        # Immediate Distance Penalty: Power-law decay
        # Higher distance reduces score via power law
        if dist_to_candidate > 1e-9:
            distance_penalty_factor = 1.0 / (1.0 + (dist_to_candidate ** alpha_dist))
        else:
            distance_penalty_factor = 1.0

        # Sigmoid Budget Factor
        # Smooth transition from exploration to conservation
        if remaining_budget > 1e-9:
            norm_budget = budget_after_first_step / remaining_budget
            norm_budget = max(0.0, min(1.0, norm_budget))

            # Sigmoid function centered at k_sigmoid
            # When norm_budget is high (near 1), factor is near 1.
            # When norm_budget is low (near 0), factor is near 0.
            sigmoid_component = 1.0 / (1.0 + np.exp(-steepness_sigmoid * (norm_budget - k_sigmoid)))

            # Normalize sigmoid output to range [0, 1] relative to inflection
            # At norm_budget=1, sigmoid ~ 1. At norm_budget=0, sigmoid ~ 0.
            # We map it such that full budget gives factor 1.0 and 0 budget gives 0.0
            max_sigmoid = 1.0 / (1.0 + np.exp(-steepness_sigmoid * (1.0 - k_sigmoid)))
            min_sigmoid = 1.0 / (1.0 + np.exp(-steepness_sigmoid * (0.0 - k_sigmoid)))

            if max_sigmoid > min_sigmoid:
                budget_factor = (sigmoid_component - min_sigmoid) / (max_sigmoid - min_sigmoid)
            else:
                budget_factor = norm_budget # Fallback to linear if sigmoid is flat

        else:
            budget_factor = 0.0

        # Calculate final score
        score = total_path_value * distance_penalty_factor * budget_factor

        if score > best_score:
            best_score = score
            best_node = candidate

    return best_node

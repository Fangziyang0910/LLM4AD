import numpy as np

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: float, current_time: float, demands: np.ndarray, distance_matrix: np.ndarray, time_windows: np.ndarray) -> int:
    """
    Selects the next node using a composite score integrating capacity-aware regret,
    adaptive urgency, and a 2-step lookahead that penalizes dead-ends.
    """
    if len(unvisited_nodes) == 0:
        return depot

    unvisited_nodes = np.asarray(unvisited_nodes)
    
    if len(unvisited_nodes) == 1:
        node = int(unvisited_nodes[0])
        if demands[node] <= rest_capacity:
            arrival = current_time + distance_matrix[current_node, node]
            if arrival <= time_windows[node, 1] + 1e-9:
                return node
        return depot

    candidates = []
    max_demand_unvisited = 0.0
    for node in unvisited_nodes:
        node_idx = int(node)
        if demands[node_idx] > rest_capacity:
            continue
        dist = distance_matrix[current_node, node_idx]
        arrival_time = current_time + dist
        tw_start = time_windows[node_idx, 0]
        tw_end = time_windows[node_idx, 1]
        if arrival_time > tw_end + 1e-9:
            continue
        
        # Track max demand among ALL unvisited to assess capacity risk
        if demands[node_idx] > max_demand_unvisited:
            max_demand_unvisited = demands[node_idx]
            
        wait_time = max(0.0, tw_start - arrival_time)
        service_start = max(arrival_time, tw_start)
        slack = tw_end - service_start
        base_cost = dist + 0.3 * wait_time

        candidates.append({
            'node': node_idx,
            'cost': base_cost,
            'slack': slack,
            'arrival_time': service_start,
            'remaining_capacity': rest_capacity - demands[node_idx],
            'demand': demands[node_idx]
        })
    
    if not candidates:
        return depot
        
    if len(candidates) == 1:
        return candidates[0]['node']

    n_cand = len(candidates)
    costs = np.array([c['cost'] for c in candidates])
    slacks = np.array([c['slack'] for c in candidates])
    nodes = np.array([c['node'] for c in candidates])
    arrivals = np.array([c['arrival_time'] for c in candidates])
    rem_caps = np.array([c['remaining_capacity'] for c in candidates])

    # 1. Normalize Costs
    min_c, max_c = np.min(costs), np.max(costs)
    cost_range = max_c - min_c if max_c > min_c + 1e-9 else 1.0
    norm_costs = (costs - min_c) / cost_range

    # 2. Adaptive Urgency
    avg_slack = np.mean(slacks)
    min_slack = np.min(slacks)
    # Dynamic lambda: tighter windows -> steeper decay
    lambda_urgency = 0.1 / (1.0 + avg_slack / 50.0)
    urgencies = np.exp(-lambda_urgency * slacks)
    
    # 3. Capacity-Aware Regret
    # We compute an effective cost that includes a penalty if picking a node
    # leaves capacity insufficient for the largest remaining demand.
    if max_demand_unvisited == 0:
        max_demand_unvisited = 1.0
        
    effective_costs = np.zeros(n_cand)
    regrets = np.zeros(n_cand)
    
    for i in range(n_cand):
        # Calculate capacity risk penalty
        rem_cap_i = rem_caps[i]
        if rem_cap_i < max_demand_unvisited:
            risk_ratio = rem_cap_i / max_demand_unvisited
            cap_risk = (1.0 - risk_ratio) ** 2
        else:
            cap_risk = 0.0
        
        # Scale risk to cost magnitude
        risk_penalty = cap_risk * cost_range * 0.5
        effective_costs[i] = costs[i] + risk_penalty
        
        # Calculate regret: difference between best alternative and current
        if n_cand > 2:
            mask = np.ones(n_cand, dtype=bool)
            mask[i] = False
            best_alt_cost = np.min(costs[mask])
        else:
            other_idx = 1 - i
            best_alt_cost = costs[other_idx]
            
        # Regret is higher if current option is much better than alternatives
        # Using effective cost for the current option to reflect risk
        regrets[i] = best_alt_cost - effective_costs[i]

    min_reg, max_reg = np.min(regrets), np.max(regrets)
    regret_range = max_reg - min_reg if max_reg > min_reg + 1e-9 else 1.0
    norm_regrets = (regrets - min_reg) / regret_range

    # 4. 2-Step Lookahead (Dead-end detection and cost estimation)
    lookahead_scores = np.zeros(n_cand)
    for i in range(n_cand):
        curr_idx = nodes[i]
        t_after = arrivals[i]
        cap_after = rem_caps[i]
        
        best_next_val = float('inf')
        has_next_option = False
        
        # Option A: Return to Depot (Safe, ends route)
        dist_depot = distance_matrix[curr_idx, depot]
        best_next_val = min(best_next_val, dist_depot)
        has_next_option = True
        
        # Option B: Go to another customer
        for j in range(n_cand):
            if i == j:
                continue
            node_j = nodes[j]
            if demands[node_j] > cap_after:
                continue
            dist_ij = distance_matrix[curr_idx, node_j]
            arr_j = t_after + dist_ij
            tw_s_j = time_windows[node_j, 0]
            tw_e_j = time_windows[node_j, 1]
            if arr_j > tw_e_j + 1e-9:
                continue
            
            wait_j = max(0.0, tw_s_j - arr_j)
            cost_j = dist_ij + 0.3 * wait_j
            if cost_j < best_next_val:
                best_next_val = cost_j
                has_next_option = True
        
        if not has_next_option:
            # This shouldn't happen because depot is always an option, 
            # but if depot distance is considered 'return', we treat it as valid.
            # However, if we want to strictly penalize dead-ends where we *must* return,
            # the depot distance is the cost. If we wanted to penalize *customers* 
            # that lead nowhere, we would check if any *customer* was reachable.
            # Here, best_next_val is at most dist_depot.
            pass
            
        # Normalize later, but store raw for now
        lookahead_scores[i] = best_next_val

    min_lk, max_lk = np.min(lookahead_scores), np.max(lookahead_scores)
    lk_range = max_lk - min_lk if max_lk > min_lk + 1e-9 else 1.0
    norm_lookaheads = (lookahead_scores - min_lk) / lk_range

    # 5. Adaptive Weights
    w_cost = 0.40
    w_urgency = 0.30
    w_regret = 0.15
    w_lookahead = 0.15

    # Adapt based on slack (time tightness)
    if min_slack < 10:
        w_urgency *= 2.0
        w_regret *= 1.5
        w_cost *= 0.8
    elif avg_slack < 20:
        w_urgency *= 1.5
        
    # Adapt based on capacity tightness
    if rest_capacity < max_demand_unvisited * 1.5:
        w_regret *= 1.5
        w_cost *= 0.9 # Prioritize efficiency/packing

    total_w = w_cost + w_urgency + w_regret + w_lookahead
    w_cost /= total_w
    w_urgency /= total_w
    w_regret /= total_w
    w_lookahead /= total_w

    # Final Score: Minimize
    # Cost: lower is better
    # Urgency: higher is better (minimize 1-urgency)
    # Regret: higher is better (minimize 1-regret)
    # Lookahead: lower next cost is better
    score = (
        w_cost * norm_costs + 
        w_urgency * (1.0 - urgencies) + 
        w_regret * (1.0 - norm_regrets) +
        w_lookahead * norm_lookaheads
    )
    
    best_idx = np.argmin(score)
    return int(nodes[best_idx])
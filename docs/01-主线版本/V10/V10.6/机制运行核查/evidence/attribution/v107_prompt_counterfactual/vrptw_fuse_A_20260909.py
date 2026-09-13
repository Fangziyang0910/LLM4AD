import numpy as np

def select_next_node(current_node: int, depot: int, unvisited_nodes: np.ndarray, rest_capacity: float, current_time: float, demands: np.ndarray, distance_matrix: np.ndarray, time_windows: np.ndarray) -> int:
    """
    Selects the next node using a composite score combining capacity-aware regret,
    2-step lookahead, and adaptive urgency.
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
    
    # Pre-calculate max demand among all unvisited nodes for capacity risk
    if len(unvisited_nodes) > 0:
        max_demand_unvisited = np.max(demands[unvisited_nodes])
    max_demand_unvisited = max(max_demand_unvisited, 1e-9)

    # Step 1: Filter feasible candidates and compute raw metrics
    for node in unvisited_nodes:
        node_idx = int(node)
        
        # Capacity Check
        if demands[node_idx] > rest_capacity:
            continue
            
        dist = distance_matrix[current_node, node_idx]
        arrival_time = current_time + dist
        
        tw_start = time_windows[node_idx, 0]
        tw_end = time_windows[node_idx, 1]
        
        # Time Window Feasibility Check
        if arrival_time > tw_end + 1e-9:
            continue
        
        wait_time = max(0.0, tw_start - arrival_time)
        service_start = max(arrival_time, tw_start)
        slack = tw_end - service_start
        
        # Base Cost: Distance + Waiting Time
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

    # Normalize costs to [0, 1]
    min_c, max_c = np.min(costs), np.max(costs)
    cost_range = max_c - min_c if max_c > min_c + 1e-9 else 1.0
    norm_costs = (costs - min_c) / cost_range

    # Calculate Adaptive Urgency
    avg_slack = np.mean(slacks)
    min_slack = np.min(slacks)
    # Dynamic lambda: tighter windows -> steeper decay
    lambda_urgency = 0.1 / (1.0 + avg_slack / 50.0)
    urgencies = np.exp(-lambda_urgency * slacks)
    
    # Calculate Capacity-Aware Regret
    # Regret_i = (Best Alternative Cost) - (Effective Cost_i)
    # Effective Cost includes a penalty if remaining capacity is insufficient for max future demand
    regrets = np.zeros(n_cand)
    for i in range(n_cand):
        if n_cand > 2:
            mask = np.ones(n_cand, dtype=bool)
            mask[i] = False
            best_alt_cost = np.min(costs[mask])
        else:
            other_idx = 1 - i
            best_alt_cost = costs[other_idx]
        
        # Capacity Risk Penalty
        rem_cap_i = rem_caps[i]
        if rem_cap_i < max_demand_unvisited:
            risk_ratio = rem_cap_i / max_demand_unvisited
            cap_risk = (1.0 - risk_ratio) ** 2
        else:
            cap_risk = 0.0
        
        # Scale risk to be comparable to cost difference
        risk_penalty = cap_risk * (cost_range) * 0.5
        
        eff_cost_i = costs[i] + risk_penalty
        regrets[i] = best_alt_cost - eff_cost_i
        
    min_reg, max_reg = np.min(regrets), np.max(regrets)
    regret_range = max_reg - min_reg if max_reg > min_reg + 1e-9 else 1.0
    norm_regrets = (regrets - min_reg) / regret_range

    # Calculate Lookahead Cost: 2-step simulation
    # Estimate the cost of the best next move after visiting candidate i
    lookahead_costs = np.zeros(n_cand)
    for i in range(n_cand):
        current_node_idx = nodes[i]
        current_time_after_visit = arrivals[i]
        current_cap_after_visit = rem_caps[i]
        
        min_next_cost = float('inf')
        
        # Option 1: Go to Depot
        dist_to_depot = distance_matrix[current_node_idx, depot]
        min_next_cost = min(min_next_cost, dist_to_depot)
        
        # Option 2: Go to another feasible customer
        for j in range(n_cand):
            if i == j:
                continue
            
            node_j = nodes[j]
            
            # Check capacity
            if demands[node_j] > current_cap_after_visit:
                continue
                
            dist_ij = distance_matrix[current_node_idx, node_j]
            arrival_j = current_time_after_visit + dist_ij
            
            tw_start_j = time_windows[node_j, 0]
            tw_end_j = time_windows[node_j, 1]
            
            if arrival_j > tw_end_j + 1e-9:
                continue
            
            wait_j = max(0.0, tw_start_j - arrival_j)
            service_start_j = max(arrival_j, tw_start_j)
            cost_j = dist_ij + 0.3 * wait_j
            
            if cost_j < min_next_cost:
                min_next_cost = cost_j
        
        if min_next_cost == float('inf'):
            min_next_cost = 0.0
            
        lookahead_costs[i] = min_next_cost

    # Normalize lookahead costs
    min_lk, max_lk = np.min(lookahead_costs), np.max(lookahead_costs)
    lk_range = max_lk - min_lk if max_lk > min_lk + 1e-9 else 1.0
    norm_lookaheads = (lookahead_costs - min_lk) / lk_range

    # Adaptive Weights
    w_cost = 0.4
    w_urgency = 0.3
    w_regret = 0.15
    w_lookahead = 0.15
    
    # Adapt weights based on slack and capacity constraints
    if min_slack < 10:
        w_urgency *= 2.0
        w_regret *= 1.5
        w_cost *= 0.8
        w_lookahead *= 0.5
    elif avg_slack < 20:
        w_urgency *= 1.5
        
    # If remaining capacity is tight relative to max demand, boost regret weight to avoid capacity traps
    if rest_capacity < max_demand_unvisited * 1.5:
        w_regret *= 1.5

    total_w = w_cost + w_urgency + w_regret + w_lookahead
    w_cost /= total_w
    w_urgency /= total_w
    w_regret /= total_w
    w_lookahead /= total_w

    # Final Score Minimization
    # 1. Cost: Lower is better
    # 2. Urgency: Higher urgency is better -> Penalize low urgency (1 - urgency)
    # 3. Regret: Higher regret is better -> Penalize low regret (1 - regret)
    # 4. Lookahead: Lower distance to next step is better
    
    score = (
        w_cost * norm_costs + 
        w_urgency * (1.0 - urgencies) + 
        w_regret * (1.0 - norm_regrets) +
        w_lookahead * norm_lookaheads
    )
    
    best_idx = np.argmin(score)
    return int(nodes[best_idx])
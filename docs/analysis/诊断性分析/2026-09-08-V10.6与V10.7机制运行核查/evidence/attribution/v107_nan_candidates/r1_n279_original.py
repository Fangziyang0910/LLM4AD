import numpy as np
from functools import lru_cache
import sys

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    """
    Design a novel algorithm to select the next node in each step.

    Args:
    current_node: ID of the current node.
    destination_node: ID of the destination node.
    unvisited_nodes: Array of IDs of unvisited nodes.
    distance_matrix: Distance matrix of nodes.

    Return:
    ID of the next node to visit.
    """
    unvisited_nodes = np.asarray(unvisited_nodes, dtype=int)
    n = len(unvisited_nodes)
    
    # Base cases
    if n == 0:
        return int(destination_node)
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Exact solution for the last 12 nodes using DP (Held-Karp style) for better end-game quality
    # 2^12 * 12 * 12 ~ 589k ops. This is fast in C-backed numpy/python if cached, but lru_cache adds overhead.
    # For n=12, states are 4096*12=49k. Transitions are ~49k*12=600k. This is acceptable per call.
    if n <= 12:
        # Map global node ids to local indices 0..n-1 for DP efficiency
        local_to_global = unvisited_nodes
        global_to_local = {int(g): i for i, g in enumerate(local_to_global)}
        
        dest_global = destination_node
        dest_in_unvisited = dest_global in global_to_local
        
        # Ensure recursion limit is high enough for DP depth (max n=12, depth is small)
        if sys.getrecursionlimit() < 1000:
            sys.setrecursionlimit(10000)
            
        if dest_in_unvisited:
            dest_local = global_to_local[dest_global]
            # We must visit all other nodes, then end at dest_local.
            # Subproblem: visit all nodes in (unvisited \ {dest}), starting from current.
            other_nodes_local = [i for i in range(n) if i != dest_local]
            m = len(other_nodes_local) # m = n-1
            
            if m == 0:
                return int(dest_global)
                
            # DP on subset of other_nodes_local.
            @lru_cache(maxsize=None)
            def dp(mask, last_idx):
                # mask: bitmask of which nodes from other_nodes_local have been visited
                # last_idx: index in other_nodes_local of the last visited node
                if mask == (1 << m) - 1:
                    global_last = local_to_global[other_nodes_local[last_idx]]
                    return distance_matrix[global_last, dest_global]
                
                best = float('inf')
                for next_idx in range(m):
                    if not (mask & (1 << next_idx)):
                        new_mask = mask | (1 << next_idx)
                        curr_global = local_to_global[other_nodes_local[last_idx]]
                        next_global = local_to_global[other_nodes_local[next_idx]]
                        cost = distance_matrix[curr_global, next_global] + dp(new_mask, next_idx)
                        if cost < best:
                            best = cost
                return best
            
            best_total = float('inf')
            best_first_node = None
            for start_idx in range(m):
                start_global = local_to_global[other_nodes_local[start_idx]]
                first_cost = distance_matrix[current_node, start_global]
                rest_cost = dp(1 << start_idx, start_idx)
                total = first_cost + rest_cost
                if total < best_total:
                    best_total = total
                    best_first_node = start_global
                    
            return int(best_first_node)

        else:
            # Destination is not in unvisited. Visit all unvisited nodes and return to destination.
            @lru_cache(maxsize=None)
            def dp(mask, last_idx):
                if mask == (1 << n) - 1:
                    global_last = local_to_global[last_idx]
                    return distance_matrix[global_last, dest_global]
                
                best = float('inf')
                for next_idx in range(n):
                    if not (mask & (1 << next_idx)):
                        new_mask = mask | (1 << next_idx)
                        curr_global = local_to_global[last_idx]
                        next_global = local_to_global[next_idx]
                        cost = distance_matrix[curr_global, next_global] + dp(new_mask, next_idx)
                        if cost < best:
                            best = cost
                return best
            
            best_total = float('inf')
            best_first_node = None
            for start_idx in range(n):
                start_global = local_to_global[start_idx]
                first_cost = distance_matrix[current_node, start_global]
                rest_cost = dp(1 << start_idx, start_idx)
                total = first_cost + rest_cost
                if total < best_total:
                    best_total = total
                    best_first_node = start_global
                    
            return int(best_first_node)

    # Heuristic for n > 12
    
    # 1. Immediate Cost: Distance from current node to each candidate
    dist_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # 2. Destination Cost: Distance from candidate to destination
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    
    # 3. Lookahead & Cluster Metrics
    sub_matrix = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    sub_matrix_no_self = sub_matrix.copy()
    np.fill_diagonal(sub_matrix_no_self, np.inf)
    
    # 3a. Two-Step Lookahead
    min_next_unvisited = np.min(sub_matrix_no_self, axis=1)
    lookahead_cost = np.minimum(min_next_unvisited, dist_to_dest)
    
    # 3b. Isolation Penalty
    avg_dist_to_cluster = np.mean(sub_matrix_no_self, axis=1)
    max_avg_cluster = np.max(avg_dist_to_cluster)
    if max_avg_cluster > 1e-9:
        norm_isolation = avg_dist_to_cluster / max_avg_cluster
    else:
        norm_isolation = np.zeros(n)
        
    # 4. Adaptive Weights
    # Reference point for "late game" in heuristic phase is n=13 (just above DP limit)
    # Reference point for "early game" is a larger number, say 50 or 100.
    # Let's use a reference max of 50 for the ramp.
    ref_n = 50
    if n <= 13:
        progress = 1.0
    else:
        # Linear ramp from 0 at ref_n to 1 at 13
        progress = np.clip(1.0 - (n - 13) / (ref_n - 13), 0.0, 1.0)
        
    # w_dest: Stronger pull as we get closer to end.
    # Quadratic ramp ensures it stays low for long tours but kicks in hard when remaining nodes are few.
    w_dest = 0.2 + 0.6 * (progress ** 2)
    
    # w_look: Lookahead weight.
    w_look = 0.4
    
    # w_iso: Isolation penalty.
    w_iso = 0.1 + 0.3 * progress
    
    # 5. Composite Score
    scores = dist_from_current + w_look * lookahead_cost + w_dest * dist_to_dest + w_iso * norm_isolation
    
    # 6. Prevent selecting destination node early
    if destination_node in unvisited_nodes:
        dest_idx = np.where(unvisited_nodes == destination_node)[0]
        if len(dest_idx) > 0:
            scores[dest_idx[0]] = float('inf')
            
    # 7. Selection
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
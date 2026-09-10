import numpy as np


def select_next_node(current_node, depot, unvisited_nodes, rest_capacity, current_time,
                     demands, distance_matrix, time_windows):
    """Time-aware nearest insertion with one-step continuation savings."""
    nodes = np.asarray(unvisited_nodes, dtype=int)
    if len(nodes) == 0:
        return int(depot)
    d = np.asarray(distance_matrix, dtype=float)
    tw = np.asarray(time_windows, dtype=float)
    travel = d[current_node, nodes]
    start = np.maximum(current_time + travel, tw[nodes, 0])
    wait = start - current_time - travel
    radial = d[depot, nodes]
    pair = d[np.ix_(nodes, nodes)]
    # Window width is a visible proxy for the unobserved service duration.
    service_proxy = float(np.median(tw[nodes, 1] - tw[nodes, 0]))
    arrival2 = start[:, None] + service_proxy + pair
    start2 = np.maximum(arrival2, tw[nodes, 0][None, :])
    feasible = (start2 <= tw[nodes, 1][None, :])
    feasible &= (demands[nodes][:, None] + demands[nodes][None, :] <= rest_capacity)
    np.fill_diagonal(feasible, False)
    savings = np.maximum(radial[:, None] + radial[None, :] - pair, 0.0)
    delay = np.maximum(start2 - arrival2, 0.0)
    continuation = np.max(np.where(feasible, savings / (1.0 + delay), 0.0), axis=1)
    score = travel + 0.15 * wait - 0.25 * radial + 0.04 * tw[nodes, 1] - 0.25 * continuation
    return int(nodes[np.argmin(score)])

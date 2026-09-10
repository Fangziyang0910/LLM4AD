import numpy as np


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """Multi-start nearest-neighbor tours improved by best-improvement 2-opt."""
    d = np.asarray(distance_matrix, dtype=float)
    remaining = np.asarray(unvisited_nodes, dtype=int)
    key = d.tobytes()
    cached = getattr(select_next_node, '_cached', None)
    if cached is not None and cached[0] == key:
        tour = cached[1]
        where = tour.index(int(current_node))
        if set(tour[where + 1:]) == set(remaining):
            return int(tour[where + 1])
    nodes = np.r_[int(current_node), remaining]
    n = len(nodes)
    # The normal first call exposes every city; build a complete cyclic tour.
    if current_node != destination_node:
        return int(remaining[np.argmin(d[current_node, remaining])])
    local = d[np.ix_(nodes, nodes)]
    ii, jj = np.indices((n, n))
    valid = (jj >= ii + 2) & ~((ii == 0) & (jj == n - 1))
    best, best_cost = None, float('inf')
    for start in np.linspace(0, n-1, min(n, 16), dtype=int):
        route = [int(start)]
        unused = np.ones(n, dtype=bool)
        unused[start] = False
        while len(route) < n:
            choices = np.flatnonzero(unused)
            nxt = int(choices[np.argmin(local[route[-1], choices])])
            route.append(nxt)
            unused[nxt] = False
        route = np.asarray(route)
        for _ in range(100):
            nxt = np.roll(route, -1)
            gain = (local[route[:, None], route[None, :]]
                    + local[nxt[:, None], nxt[None, :]]
                    - local[route, nxt][:, None] - local[route, nxt][None, :])
            gain[~valid] = np.inf
            i, j = np.unravel_index(np.argmin(gain), gain.shape)
            if gain[i, j] >= -1e-12:
                previous = np.roll(route, 1)
                removal = local[previous, nxt] - local[previous, route] - local[route, nxt]
                relocation = (removal[:, None] + local[route[:, None], route[None, :]]
                              + local[route[:, None], nxt[None, :]] - local[route, nxt][None, :])
                relocation[(jj == ii) | (jj == (ii-1) % n)] = np.inf
                u, edge = np.unravel_index(np.argmin(relocation), relocation.shape)
                if relocation[u, edge] >= -1e-12:
                    break
                moved, anchor = int(route[u]), int(route[edge])
                reordered = route.tolist()
                reordered.remove(moved)
                reordered.insert(reordered.index(anchor)+1, moved)
                route = np.asarray(reordered)
                continue
            route[i+1:j+1] = route[i+1:j+1][::-1]
        cost = float(local[route, np.roll(route, -1)].sum())
        if cost < best_cost:
            best_cost, best = cost, route.copy()
    tour = nodes[best].tolist()
    offset = tour.index(int(destination_node))
    tour = tour[offset:] + tour[:offset]
    select_next_node._cached = (key, tour)
    return int(tour[1])

import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """Construct capacity-feasible savings routes, then encode their edges."""
    d = np.asarray(distance_matrix, dtype=float).copy()
    np.fill_diagonal(d, 0.0)
    n = len(d)
    best_routes, best_cost = None, float('inf')
    pairs = [(i, j) for i in range(1, n) for j in range(i+1, n)]
    rng = np.random.default_rng(0)
    for trial in range(12):
        factor = (0.6, 1.0, 1.4)[trial % 3]
        merit = np.array([d[0,i]+d[0,j]-factor*d[i,j] for i,j in pairs])
        if trial >= 3:
            merit *= rng.uniform(0.85, 1.15, len(pairs))
        routes = {i: [i] for i in range(1, n)}
        owner = np.arange(n)
        loads = {i: float(demands[i]) for i in range(1, n)}
        for k in np.argsort(-merit):
            i, j = pairs[k]
            a, b = int(owner[i]), int(owner[j])
            if a == b or loads[a]+loads[b] > capacity:
                continue
            ra, rb = routes[a], routes[b]
            if i not in (ra[0], ra[-1]) or j not in (rb[0], rb[-1]):
                continue
            if ra[0] == i:
                ra = ra[::-1]
            if rb[-1] == j:
                rb = rb[::-1]
            routes[a] = ra+rb
            loads[a] += loads.pop(b)
            for node in rb:
                owner[node] = a
            del routes[b]
        optimized = []
        total = 0.0
        for customers in routes.values():
            route = [0]+customers+[0]
            for _ in range(30):
                gain, move = 0.0, None
                for i in range(1, len(route)-2):
                    for j in range(i+1, len(route)-1):
                        delta = d[route[i-1],route[j]]+d[route[i],route[j+1]]-d[route[i-1],route[i]]-d[route[j],route[j+1]]
                        if delta < gain-1e-12:
                            gain, move = delta, (i,j)
                if move is None:
                    break
                i,j = move
                route[i:j+1] = route[i:j+1][::-1]
            total += sum(d[a,b] for a,b in zip(route,route[1:]))
            optimized.append(route)
        if total < best_cost:
            best_cost, best_routes = total, optimized
    prior = 1e-4 / np.maximum(d, 1e-6)
    for route in best_routes:
        for a,b in zip(route,route[1:]):
            prior[a,b] = 1e3 / max(d[a,b],1e-6)
    np.fill_diagonal(prior, 1e-9)
    return prior

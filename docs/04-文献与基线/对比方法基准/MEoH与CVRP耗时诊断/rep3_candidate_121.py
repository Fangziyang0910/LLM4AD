import numpy as np


def heuristics(
        distance_matrix: np.ndarray,
        coordinates: np.ndarray,
        demands: np.ndarray,
        capacity: int,
) -> np.ndarray:
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.full((n, n), 1e-9, dtype=float)

    d = np.asarray(distance_matrix, dtype=float)
    d = np.where(np.isfinite(d), d, np.inf)
    d = np.maximum(d, 1e-9)
    np.fill_diagonal(d, np.inf)

    # Base: inverse distance (short edges preferred)
    base = 1.0 / d
    np.fill_diagonal(base, 0.0)

    customers = np.arange(1, n)
    if customers.size == 0:
        base = np.where(np.isfinite(base), base, 1e-9)
        base = np.where(base <= 0.0, 1e-9, base)
        return base

    cap = float(capacity) if capacity is not None and capacity > 0 else 1e9
    dem = np.zeros(n, dtype=float)
    if demands is not None and len(demands) >= n:
        dem = np.asarray(demands, dtype=float)
        dem = np.where(np.isfinite(dem), dem, 0.0)
        dem = np.maximum(dem, 0.0)
    dem[0] = 0.0

    total_dem = float(dem[1:].sum())
    # Fleet tightness: 0 = very loose, 1 = very tight
    tightness = min(1.0, total_dem / max(cap * max(n - 1, 1), 1e-9))

    # --- Phase 1: Greedily form capacity-feasible route segments ---
    # Start a new route at the farthest remaining customer (spread across map),
    # then greedily extend by nearest feasible neighbor until capacity is hit.
    visited = np.zeros(n, dtype=bool)
    visited[0] = True  # depot not a "customer" to cluster
    remaining = customers.copy()
    segments = []
    while remaining.size > 0:
        # Pick a seed: customer with max distance to its nearest unvisited neighbor
        # (encourages spreading seeds across the spatial domain)
        if remaining.size == 1:
            seed = int(remaining[0])
        else:
            # For each remaining customer, compute min distance to other remaining
            min_d = np.full(remaining.size, np.inf)
            for idx, c in enumerate(remaining):
                dd = d[c, remaining]
                dd[idx] = np.inf
                min_d[idx] = np.min(dd)
            seed = int(remaining[np.argmax(min_d)])

        route = [seed]
        visited[seed] = True
        load = float(dem[seed])
        last = seed
        while load < cap - 1e-12:
            # Find nearest unvisited customer that fits
            best_j = -1
            best_dist = np.inf
            for j in customers:
                if visited[j]:
                    continue
                if load + float(dem[j]) > cap + 1e-12:
                    continue
                dist = d[last, j]
                if dist < best_dist:
                    best_dist = dist
                    best_j = int(j)
            if best_j < 0:
                break
            visited[best_j] = True
            route.append(best_j)
            load += float(dem[best_j])
            last = best_j
        segments.append((route, load))

    # --- Build edge-prior from segments ---
    # Intra-route directed edges: strong boost (route coherence)
    # Depot-to-route-start: moderate boost (fresh route starts)
    # Route-end-to-depot: boosted when route is well-filled
    # Customer-to-depot (non-route-end): suppressed
    # Depot-to-non-route-start: mildly boosted

    mult = np.ones((n, n), dtype=float)

    # Intra-route edges
    for route, load in segments:
        for k in range(len(route) - 1):
            u, v = route[k], route[k + 1]
            mult[u, v] *= 4.0

    # Depot to route start (fresh route)
    for route, load in segments:
        if len(route) == 0:
            continue
        s = route[0]
        mult[0, s] *= 2.5

    # Route end to depot (return trip)
    for route, load in segments:
        if len(route) == 0:
            continue
        e = route[-1]
        fill_ratio = min(1.0, load / cap) if cap < 1e9 else 0.5
        # Boost return more when route is well-filled
        mult[e, 0] *= 1.0 + 2.0 * fill_ratio

    # Suppress direct customer-to-depot for non-end customers (force route completion)
    for route, load in segments:
        for k in range(len(route) - 1):
            c = route[k]
            mult[c, 0] *= 0.05

    # Depot to non-start customers: mildly boosted (alternate route starts)
    start_set = set()
    for route, load in segments:
        if route:
            start_set.add(route[0])
    for j in customers:
        if j not in start_set:
            mult[0, j] *= 1.3

    # --- Phase 2: Fleet load imbalance modulation ---
    # Customers in under-filled segments are more attractive to fresh routes
    # Customers in nearly-full segments are less attractive for entry
    segment_load = {}
    for idx, (route, load) in enumerate(segments):
        for c in route:
            segment_load[c] = load

    avg_load = np.mean([s[1] for s in segments]) if segments else 0.0
    for j in customers:
        load_j = segment_load.get(j, 0.0)
        # Imbalance: 0 if average, 1 if very under/over filled relative to cap
        imbalance = abs(load_j - cap * 0.6) / cap if cap < 1e9 else 0.0
        imbalance = min(1.0, imbalance)
        # Under-filled routes: customers more attractive (need more to fill)
        # Over-filled: less attractive
        if load_j < cap * 0.6:
            mult[:, j] *= 1.0 + 1.2 * imbalance
        else:
            mult[:, j] *= 1.0 - 0.5 * imbalance

    # Apply modulation to base
    base = base * mult

    # Depot suppression: entering depot should be rarer than leaving
    base[1:, 0] *= 0.15
    base[0, 1:] *= 1.1

    # Ensure finite and positive
    base = np.where(np.isfinite(base), base, 1e-9)
    base = np.where(base <= 0.0, 1e-9, base)

    return base

import numpy as np


def _nn_routes(dem, cap, d, starts):
    """Capacity-feasible nearest-neighbor tours starting from given customers.

    Returns a list of routes (each as a list of node indices starting and
    ending at depot 0). Each customer is assigned to at most one route.
    """
    n = d.shape[0]
    if n <= 1 or cap <= 0 or cap >= 1e9:
        return []
    used = set()
    routes = []
    for s in starts:
        if not (0 < s < n) or s in used:
            continue
        used.add(s)
        route = [0, s]
        load = float(dem[s])
        cur = s
        while True:
            # Find nearest feasible (capacity) unused customer.
            best = -1
            best_d = np.inf
            for j in range(1, n):
                if j in used or j == s:
                    continue
                if load + float(dem[j]) > cap + 1e-12:
                    continue
                dj = d[cur, j]
                if np.isfinite(dj) and dj < best_d:
                    best_d = dj
                    best = j
            if best < 0:
                break
            route.append(best)
            used.add(best)
            load += float(dem[best])
            cur = best
        route.append(0)
        routes.append(route)
    return routes


def _two_opt_once(route, d):
    """One bounded 2-opt pass to reduce in-route crossings."""
    if len(route) <= 3:
        return list(route)
    r = list(route)
    improved = True
    it = 0
    while improved and it < 3:
        improved = False
        it += 1
        m = len(r)
        for i in range(1, m - 2):
            for j in range(i + 1, m - 1):
                d1 = d[r[i - 1], r[i]] + d[r[j], r[j + 1]]
                d2 = d[r[i - 1], r[j]] + d[r[i], r[j + 1]]
                if d2 + 1e-12 < d1:
                    r[i:j + 1] = r[i:j + 1][::-1]
                    improved = True
    return r


def heuristics(
        distance_matrix: np.ndarray,
        coordinates: np.ndarray,
        demands: np.ndarray,
        capacity: int,
) -> np.ndarray:
    """Return edge desirability values for CVRP ant colony optimization."""
    n = distance_matrix.shape[0]
    if n <= 1:
        return np.full((n, n), 1e-9, dtype=float)

    d = np.asarray(distance_matrix, dtype=float)
    d = np.where(np.isfinite(d), d, np.inf)
    d = np.maximum(d, 1e-9)
    np.fill_diagonal(d, np.inf)

    # Inverse-distance base: closer directed edges are more attractive.
    base = 1.0 / d
    np.fill_diagonal(base, 0.0)
    # Suppress direct customer -> depot returns so ants prefer longer, fuller routes.
    base[1:, 0] *= 0.05
    # Mild emphasis on leaving the depot.
    base[0, 1:] *= 1.05

    cap = float(capacity) if capacity is not None else 0.0
    if cap <= 0:
        cap = 1e9

    if demands is not None and len(demands) >= n:
        dem = np.asarray(demands, dtype=float)
        dem = np.where(np.isfinite(dem), dem, 0.0)
        dem = np.maximum(dem, 0.0)
    else:
        dem = np.zeros(n, dtype=float)

    # Capacity-aware successor landscape: distance plus a gap penalty that
    # discourages edges that would make the remaining load infeasibly heavy.
    dem_i = dem[:, None]
    dem_j = dem[None, :]
    rem_after = cap - dem_i - dem_j
    gap = np.where(rem_after < 0, -rem_after, 0.0)
    gap_score = 5.0 * gap / max(cap, 1e-9)
    score = d + gap_score
    np.fill_diagonal(score, np.inf)
    infeasible = rem_after < -1e-12
    score = np.where(infeasible, np.inf, score)

    # Soft-temperature softmax over penalized distances per source node.
    finite_mask = np.isfinite(score)
    min_score = np.full(n, np.inf)
    max_score = np.full(n, -np.inf)
    np.minimum.at(min_score, np.where(finite_mask)[0], score[finite_mask])
    np.maximum.at(max_score, np.where(finite_mask)[0], score[finite_mask])
    span = max_score - min_score
    span = np.where(np.isfinite(span) & (span > 1e-9), span, np.inf)
    temperature = np.where(np.isfinite(span), span / 8.0, 0.0)

    t = temperature[:, None]
    normed = np.where(
        (t > 1e-12) & finite_mask,
        (score - min_score[:, None]) / t,
        np.where(finite_mask, 0.0, np.inf),
    )
    normed = np.minimum(normed, 50.0)
    w = np.exp(-normed)
    w = np.where(finite_mask, w, 0.0)
    w = np.where(np.isfinite(w), w, 0.0)
    np.fill_diagonal(w, 0.0)
    w[1:, 0] = 0.0

    row_sum = np.sum(w, axis=1, keepdims=True)
    probs = w / np.where(row_sum > 0, row_sum, 1.0)
    probs = np.where(np.isfinite(probs), probs, 0.0)
    np.fill_diagonal(probs, 0.0)
    probs[1:, 0] = 0.0

    # Gently reward edges that bring a route closer to capacity (concave margin).
    margin = np.clip(rem_after, 0.0, cap)
    fill = margin / max(cap, 1e-9)
    margin_boost = 1.0 + 0.15 * (1.0 - np.sqrt(np.clip(fill, 0.0, 1.0)))
    base = base * probs * margin_boost

    # Diversified NN tour ensemble: inject coherent global arc direction via a
    # damped directed edge boost. Kept lightweight (few starts, one 2-opt pass).
    try:
        customers = list(range(1, n))
        starts = None
        if customers and cap < 1e9:
            # Pick up to 4 diverse start customers (first, and spread across the rest).
            k = max(1, min(4, len(customers)))
            if len(customers) <= k:
                starts = customers
            else:
                idx = np.linspace(0, len(customers) - 1, k).astype(int)
                starts = [customers[i] for i in idx]
        tour_boost = np.zeros((n, n), dtype=float)
        for rt in _nn_routes(dem, cap, d, starts=starts):
            rt = _two_opt_once(rt, d)
            for a, b in zip(rt[:-1], rt[1:]):
                if 0 <= a < n and 0 <= b < n and a != b and np.isfinite(base[a, b]):
                    tour_boost[a, b] += 1.0
        if tour_boost.max() > 0:
            # Damped amplification: strong on tour edges, mild off-tour.
            base *= (1.0 + 0.5 * np.sqrt(tour_boost))
    except Exception:
        pass

    base = np.where(np.isfinite(base), base, 1e-9)
    base = np.where(base <= 0.0, 1e-9, base)
    return base

import numpy as np

def _build_edges(dem, cap, d, n):
    """Capacity-aware nearest + nearest-insertion. Returns directed edge multiset."""
    edges = []
    if n <= 1 or cap <= 0:
        return edges
    customers = list(range(1, n))
    while customers:
        route = [0]
        load = 0.0
        rem = list(customers)
        # seed with nearest feasible
        feas = [j for j in rem if load + float(dem[j]) <= cap + 1e-12]
        if not feas:
            break
        nxt = min(feas, key=lambda j: d[0, j])
        edges.append((0, nxt))
        route.append(nxt)
        load += float(dem[nxt])
        rem.remove(nxt)
        # nearest insertion into the open route (between depot and end)
        changed = True
        while changed:
            changed = False
            # candidate insertion positions: after depot (index0) up to last element
            # route = [0, a1, ..., ak]; next customer j inserts at position p (1..k)
            if not rem:
                break
            best = None  # (cost, j, p)
            for j in rem:
                if load + float(dem[j]) > cap + 1e-12:
                    continue
                k = len(route) - 1  # number of internal customers
                # insert after depot (p=0) or after each of k customers
                for p in range(0, k + 1):
                    a = route[p]
                    b = route[p + 1] if p < k else 0
                    cost = d[a, j] + d[j, b] - d[a, b]
                    if best is None or cost < best[0]:
                        best = (cost, j, p)
            if best is None:
                break
            cost, j, p = best
            edges.append((route[p], j))
            if p < k:
                edges.append((j, route[p + 1]))
            route.insert(p + 1, j)
            load += float(dem[j])
            rem.remove(j)
            changed = True
        edges.append((route[-1], 0))
    return edges

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

    base = 1.0 / d
    np.fill_diagonal(base, 0.0)

    cap = float(capacity) if capacity is not None else 0.0
    if cap <= 0:
        cap = 1e9

    if demands is not None and len(demands) >= n:
        dem = np.asarray(demands, dtype=float)
        dem = np.where(np.isfinite(dem), dem, 0.0)
        dem = np.maximum(dem, 0.0)
    else:
        dem = np.zeros(n, dtype=float)

    if n > 1:
        customers = np.arange(1, n)
        off_diag = d[~np.eye(n, dtype=bool)]
        scale = max(float(np.median(off_diag)), 1e-9)

        # 1) Full per-node distance-rank weighting over feasible successors
        rank_boost = np.zeros((n, n), dtype=float)
        for i in range(n):
            rem = cap - float(dem[i])
            feas = dem[1:] <= (rem - 1e-12)
            cand = customers[feas]
            if cand.size == 0:
                continue
            dists = d[i, cand]
            order = np.argsort(dists, kind="stable")
            c = cand[order]
            dd = np.where(np.isfinite(dists[order]), dists[order], np.inf)
            m = np.where(np.isfinite(dd), dd / scale, np.inf)
            closeness = 1.0 / (1.0 + m)          # short hops favored
            rank_decay = 1.0 / (1.0 + 0.35 * np.arange(len(m), dtype=float))
            margin = (rem - dem[c]) / cap
            margin = np.clip(margin, 0.0, 1.0)
            margin_w = 1.0 + 1.0 * margin         # prefer keeping room (balance)
            w = closeness * rank_decay * margin_w
            rank_boost[i, c] += w
        rb_max = float(rank_boost.max()) if rank_boost.size else 0.0
        if rb_max > 0.0:
            rank_boost = rank_boost / rb_max
        rank_boost = np.clip(1.0 + 1.5 * rank_boost, 1.0, 4.0)

        # 2) Multi-route frequency map from nearest + insertion (diverse, robust)
        freq = np.zeros((n, n), dtype=float)
        for (a, b) in _build_edges(dem, cap, d, n):
            if 0 <= a < n and 0 <= b < n and a != b:
                freq[a, b] += 1.0
        fmax = float(freq.max()) if freq.size else 0.0
        if fmax > 0.0:
            freq = freq / fmax
        freq_mult = np.where(freq > 0.0, 1.0 + 1.2 * freq, 1.0)
        # strong emphasis on closing a route to depot (feasibility cue)
        freq_mult[:, 0] = np.where(freq[:, 0] > 0.0, 1.0 + 2.0 * freq[:, 0], 1.0)

        base = base * rank_boost * freq_mult

    base = np.where(np.isfinite(base), base, 1e-9)
    base = np.where(base <= 0.0, 1e-9, base)
    return base

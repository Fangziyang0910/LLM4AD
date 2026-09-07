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
    d = np.where(np.eye(n, dtype=bool), np.inf, d)

    # Base prior: prefer short hops.
    base = 1.0 / d
    np.fill_diagonal(base, 0.0)

    # Mild depot asymmetry: returning to depot is less attractive than leaving it.
    base[1:, 0] *= 0.05
    base[0, 1:] *= 1.05

    # Demand / capacity shaping.
    if demands is not None and len(demands) >= n:
        dem = np.asarray(demands, dtype=float)
        dem = np.where(np.isfinite(dem), dem, 0.0)
        dem = np.maximum(dem, 0.0)
    else:
        dem = np.zeros(n, dtype=float)

    cap = float(capacity) if capacity is not None else 0.0
    if cap <= 0:
        cap = 1e9

    customers = list(range(1, n))
    dcust = dem[1:]
    total_dem = float(dcust.sum()) if dcust.size else 0.0
    max_d = float(dcust.max()) if dcust.size else 0.0
    tight = min(1.0, total_dem / cap) if cap > 0 else 0.0

    # Destination desirability: lightly penalize entering large customers when routes are tight.
    if max_d > 0 and tight > 0:
        norm = dcust / max_d
        w_dest = 1.0 + 0.2 * tight * norm
        base[1:, 1:] = base[1:, 1:] / w_dest[None, :]

    boost = np.zeros((n, n), dtype=float)

    # Vectorized nearest-feasible-successor boost per starting load.
    # For each plausible current load, build a mask of feasible customers, then take the
    # nearest feasible customer from every node and amplify that edge.
    if customers:
        starts = [0.0]
        if max_d > 0:
            q = np.linspace(0.0, 0.9 * cap, min(4, max(1, 3)))
            starts.extend(q.tolist())
        # Always include a few customer loads for mid-route context.
        cust_loads = [float(dem[c]) for c in customers[:5]]
        starts.extend(cust_loads)

        for cur_load in starts:
            rem = cap - cur_load
            if rem < 0:
                continue
            feasible = np.zeros(n, dtype=bool)
            feasible[1:] = dem[1:] <= rem
            if not feasible[1:].any():
                continue
            mask = np.where(feasible[None, :], 0.0, np.inf)
            dm = d + mask
            np.fill_diagonal(dm, np.inf)
            dm[0, 0] = np.inf

            best = np.argmin(dm, axis=1)
            bd = dm[np.arange(n), best]
            ok = np.isfinite(bd) & (best >= 0)
            # Only nodes with at least one feasible successor contribute.
            rows = np.where(ok)[0]
            cols = best[rows]
            boost[rows, cols] += 1.0 / (1.0 + bd[rows])  # stronger for very short feasible hops

    # Small ensemble of capacity-feasible nearest-neighbor tours for coherent route direction.
    if customers:
        starts = customers[: min(3, len(customers))]
        for start in starts:
            visited = {0, start}
            cur = start
            load = float(dem[start])
            while len(visited) < n:
                # Find nearest unvisited customer that fits in remaining capacity.
                pool = [c for c in customers if c not in visited]
                best_k, best_dval = -1, np.inf
                for k in pool:
                    if float(dem[k]) <= cap - load:
                        dk = d[cur, k]
                        if np.isfinite(dk) and dk < best_dval:
                            best_dval = dk
                            best_k = k
                if best_k == -1:
                    # Route full or stuck: return to depot and start fresh.
                    if load >= cap or best_k == -1:
                        boost[cur, 0] += 0.0  # depot return intentionally not boosted
                        cur = 0
                        load = 0.0
                        visited = {0}
                        continue
                boost[cur, best_k] += 1.0 / (1.0 + best_dval)
                visited.add(best_k)
                cur = best_k
                load += float(dem[best_k])
                if load >= cap:
                    cur = 0
                    load = 0.0
                    visited = {0}

    # Combine: base prior multiplied by a smooth boost factor.
    boost = 1.0 + boost
    base = base * boost

    # Ensure finiteness and positivity.
    base = np.where(np.isfinite(base), base, 1e-9)
    base = np.where(base <= 0.0, 1e-9, base)
    return base

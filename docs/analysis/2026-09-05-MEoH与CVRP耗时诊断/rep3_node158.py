import numpy as np


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

    base = 1.0 / d
    np.fill_diagonal(base, 0.0)

    # Depot bias: start fresh routes, strongly suppress early return to depot.
    base[0, 1:] *= 1.08
    base[1:, 0] *= 0.03

    cap = float(capacity) if capacity is not None else 0.0
    if demands is not None and len(demands) >= n:
        dem = np.asarray(demands, dtype=float)
        dem = np.where(np.isfinite(dem), dem, 0.0)
        dem = np.maximum(dem, 0.0)
    else:
        dem = np.zeros(n, dtype=float)

    if cap <= 0:
        cap = 1e9

    if cap < 1e9 and n > 1:
        customers = np.arange(1, n)
        off_diag = d[~np.eye(n, dtype=bool)]
        finite_mask = np.isfinite(off_diag)
        scale = float(np.median(off_diag[finite_mask])) if finite_mask.any() else 1.0

        # Pairwise "load after taking edge i -> j" margin (capacity awareness).
        dem_i = dem[:, None]
        dem_j = dem[None, :]
        rem_after = cap - dem_i - dem_j
        feas = (rem_after >= -1e-12)
        feas[:, 0] = False
        feas[0, 0] = False

        margin = np.clip(rem_after / max(cap, 1e-9), 0.0, 1.0)

        # Fill weight: reward filling a nearly-full route (low margin).
        fill_w = np.where(feas, 0.8 + 0.7 * (1.0 - margin), 1e-9)

        # Fresh weight: front-load large customers onto fresh (low-load) routes.
        fresh_w = np.where(feas, 1.0 + 0.5 * (1.0 - dem_i / max(cap, 1e-9)), 1e-9)

        base = base * fill_w * fresh_w

        # Capacity-feasible per-node successor rank, blended with distance.
        score = d.copy()
        score = np.where(feas, d, np.inf)
        score[:, 0] = np.inf
        np.fill_diagonal(score, np.inf)

        ranks = np.full((n, n), float(n + 1), dtype=float)
        for i in range(n):
            row = score[i]
            finite_idx = np.where(np.isfinite(row))[0]
            if finite_idx.size > 0:
                row_f = row[finite_idx]
                order = np.argsort(row_f, kind="stable")
                idx_sorted = finite_idx[order]
                for k, idx in enumerate(idx_sorted):
                    ranks[i, idx] = float(k + 1)

        # Smooth exponential rank decay; infeasible edges fall back to no boost.
        alpha = 0.22
        mult = np.exp(-alpha * (ranks - 1.0))
        mult = np.where(feas, mult, 1.0)
        mult = np.where(mult > 0, mult, 1e-6)
        base = base * mult

    base = np.where(np.isfinite(base), base, 1e-9)
    base = np.where(base <= 0.0, 1e-9, base)
    return base

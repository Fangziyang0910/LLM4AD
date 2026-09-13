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

    # Inverse-distance base: shorter edges are more attractive.
    base = 1.0 / d
    np.fill_diagonal(base, 0.0)

    cap = float(capacity) if capacity is not None else 0.0
    if cap <= 0:
        cap = 1e12

    dem = np.zeros(n, dtype=float)
    if demands is not None and len(demands) >= n:
        dem = np.asarray(demands[:n], dtype=float)
        dem = np.where(np.isfinite(dem), dem, 0.0)
        dem = np.maximum(dem, 0.0)

    # Soft capacity-gap: gently discourage edges whose endpoint demand would
    # overflow the shared vehicle capacity, scaling the penalty by distance.
    gap = np.maximum(dem[None, :] - cap, 0.0)
    gap_factor = 1.0 / (1.0 + 2.0 * gap * d / cap)

    # Gentle rank decay over nearest successors keeps most edges competitive.
    ranks = np.full((n, n), n + 1, dtype=float)
    for i in range(n):
        row = d[i]
        finite_idx = np.where(np.isfinite(row))[0]
        if finite_idx.size > 0:
            order = np.argsort(row[finite_idx], kind="stable")
            ranks[i, finite_idx[order]] = np.arange(1, finite_idx.size + 1)
    rank_w = np.exp(-0.15 * (ranks - 1.0))
    rank_w = np.where(np.isfinite(rank_w), rank_w, 1e-6)

    base = base * gap_factor * rank_w
    base = np.where(np.isfinite(base), base, 1e-9)
    base = np.where(base <= 0.0, 1e-9, base)
    return base

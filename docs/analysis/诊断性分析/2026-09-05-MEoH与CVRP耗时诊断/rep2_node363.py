import numpy as np


def heuristics(
        distance_matrix: np.ndarray,
        coordinates: np.ndarray,
        demands: np.ndarray,
        capacity: int,
) -> np.ndarray:
    """Return edge desirability values for CVRP ant colony optimization."""
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])

    floor = 1e-9
    eps = 1e-12

    safe_d = np.maximum(distance_matrix, eps)

    # Sharp geometric core: favor short local moves so ants stay in dense regions.
    heuristic = 1.0 / (safe_d ** 6.0)
    np.fill_diagonal(heuristic, 0.0)

    if n > 1:
        cust = demands[1:]
        cap = float(max(int(capacity), 1))
        if len(cust) > 0 and cap > 0:
            d = cust / cap
            mean_d = max(float(np.mean(d)), 1e-12)

            cust_coords = coordinates[1:]
            pair_d = np.linalg.norm(
                cust_coords[:, None, :] - cust_coords[None, :, :], axis=2
            )
            np.fill_diagonal(pair_d, np.inf)

            # Adaptive spatial scale: mean nearest-neighbor distance (lower bounded
            # by a fraction of the typical inter-customer spacing) so the gate
            # matches the instance's true density instead of a fixed scale.
            nn = np.min(pair_d, axis=1)
            nn = nn[np.isfinite(nn)]
            if nn.size > 0:
                sigma = float(np.mean(nn))
            else:
                finite = pair_d[np.isfinite(pair_d)]
                sigma = 0.5 * max(float(np.median(finite)), 1e-12) if finite.size else 1e-12
            # Ensure the packing gate is broad enough to pair nearby customers.
            sigma = max(sigma, 0.35 * mean_d if (2.0 * mean_d > 0) else sigma, 1e-12)

            # Demand-fill: how efficiently this pair loads a vehicle.
            pair_fill = d[None, :] + d[:, None]
            fill_ratio = np.clip(pair_fill / (2.0 * mean_d), 0.0, 2.0)
            # Favor pairs that are close to (but not over) a full load.
            fill_gate = np.exp(-0.5 * (np.abs(fill_ratio - 1.0) / 1.2) ** 2)

            # Spatial gate: only reward pairs that are geometrically compact.
            spatial = np.exp(-(pair_d ** 2) / (2.0 * sigma ** 2))

            # Combined capacity-aware boost for customer-customer edges only.
            pack_boost = 1.0 + 0.75 * fill_gate * spatial
            pack_boost = np.clip(pack_boost, 1.0, 2.0)

            h = np.zeros((n, n), dtype=float)
            h[1:, 1:] = heuristic[1:, 1:] * pack_boost

            # Mild asymmetric depot bias: help ants close routes near natural
            # capacity boundaries without creating long detours.
            h[1:, 0] = h[1:, 0] * 1.08
            h[0, 1:] = h[0, 1:] * 0.96

            heuristic = h

    heuristic = np.where(
        np.isfinite(heuristic) & (heuristic > 0.0), heuristic, floor
    )
    heuristic = np.maximum(heuristic, floor)
    return heuristic

import numpy as np


def heuristics(prize, distance, maxlen):
    """Prize efficiency with neighborhood value and return-budget awareness."""
    d = np.maximum(np.asarray(distance, dtype=float), 1e-6)
    p = np.maximum(np.asarray(prize, dtype=float), 1e-8)
    neighbors = d.copy()
    np.fill_diagonal(neighbors, np.inf)
    idx = np.argsort(neighbors, axis=1)[:, :min(6, len(p)-1)]
    density = np.sum(p[idx] / np.maximum(np.take_along_axis(neighbors, idx, axis=1), 0.02), axis=1)
    density /= max(float(np.mean(density)), 1e-8)
    radial = np.asarray(distance, dtype=float)[0].copy()
    radial[0] = 0.0
    detour = radial[:, None] + d + radial[None, :]
    prior = p[None, :] ** 1.7 / d ** 1.8
    prior *= (0.5 + density[None, :]) ** 0.4
    prior *= np.exp(-0.4 * np.minimum(detour / max(float(maxlen), 1e-6), 100))
    prior[:, 0] = 1e-8
    np.fill_diagonal(prior, 1e-9)
    return np.maximum(prior, 1e-9)

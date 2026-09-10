import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """Distance, normalized depot savings, and a mild demand preference."""
    d = np.maximum(np.asarray(distance_matrix, dtype=float), 1e-6)
    radial = np.asarray(distance_matrix, dtype=float)[0]
    savings = np.maximum(radial[:, None] + radial[None, :] - d, 0.0)
    relative_savings = savings / np.maximum(radial[:, None] + radial[None, :], 1e-6)
    prior = d ** -2.5 * (0.5 + relative_savings) ** 1.5
    prior *= 1.0 + 0.4 * np.asarray(demands, dtype=float)[None, :] / max(float(capacity), 1.0)
    prior[1:, 0] *= 0.15
    np.fill_diagonal(prior, 1e-9)
    return np.maximum(prior, 1e-9)

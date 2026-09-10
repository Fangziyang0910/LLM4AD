import numpy as np


def heuristics(prize, distance, maxlen):
    """Concentrate ACO on high prize per unit distance."""
    d = np.maximum(np.asarray(distance, dtype=float), 1e-6)
    p = np.maximum(np.asarray(prize, dtype=float), 1e-8)
    prior = (p[None, :] / d) ** 2.0
    prior[:, 0] = 1e-9
    np.fill_diagonal(prior, 1e-9)
    return np.maximum(prior, 1e-9)

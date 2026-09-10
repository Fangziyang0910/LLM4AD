import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """A sharply local distance prior; let ACO handle capacity and depot returns."""
    d = np.maximum(np.asarray(distance_matrix, dtype=float), 1e-6)
    prior = d ** -3.0
    np.fill_diagonal(prior, 1e-9)
    return prior

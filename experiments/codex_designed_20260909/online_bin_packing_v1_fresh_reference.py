import numpy as np


def priority(item, bins):
    """Tight placement with a smooth penalty for small nonzero fragments."""
    bins = np.asarray(bins, dtype=float)
    remainder = np.maximum(bins - float(item), 0.0)
    scale = max(float(item), 1.0)
    score = -remainder / scale - 1.5 * np.exp(-remainder / (0.15 * scale))
    score += 5.0 * (remainder < 1e-9)
    # Equal maximum capacities are the available empty-bin candidates.
    score -= 2.0 * ((bins == np.max(bins)) & (remainder > 0))
    return score

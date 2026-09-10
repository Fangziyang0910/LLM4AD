import numpy as np


def priority(item, bins):
    """Preserve useful residual space while strongly preferring existing bins."""
    b = np.asarray(bins, dtype=float)
    r = np.maximum(b-float(item), 0.0)
    penalty = np.where((r > 0) & (r < item), 8.0*r*r/max(float(item),1.0), 0.0)
    score = -r-penalty
    score -= 20.0*np.max(b)*(b == np.max(b))
    score += float(item)*(r == 0)
    return score

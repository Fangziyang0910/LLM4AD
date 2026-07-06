template_program = '''
import numpy as np


def compute_weights(n: int) -> np.ndarray:
    """Compute observation weights for Optuna TPE's good group.

    Args:
        n: Number of observations in the good group. Observations are ordered
           from worst to best within that group.

    Returns:
        Non-negative weight array with shape (n,). Optuna normalizes weights.
    """
    if n == 0:
        return np.array([])
    if n < 25:
        return np.ones(n)
    ramp = np.linspace(1.0 / n, 1.0, num=n - 25)
    flat = np.ones(25)
    return np.concatenate([ramp, flat])
'''

task_description = (
    "Design the observation-weighting function for Optuna's Tree-structured "
    "Parzen Estimator sampler. TPE splits completed trials into good and bad "
    "groups, then fits density estimators. The designed function receives the "
    "number of observations in the good group and returns non-negative weights "
    "for those observations, ordered worst-to-best. Performance is measured by "
    "negative mean log1p best objective value across fixed one-dimensional "
    "optimization benchmarks."
)

template_program = '''
import numpy as np


def update_covariance(
    C: np.ndarray,
    p_c: np.ndarray,
    weights: np.ndarray,
    y_k: np.ndarray,
    c1: float,
    cmu: float,
    cc: float,
    hsig: float,
    n: int,
) -> np.ndarray:
    """Return the next CMA-ES covariance matrix.

    Args:
        C: current covariance matrix with shape (n, n).
        p_c: covariance evolution path.
        weights: positive recombination weights for selected offspring.
        y_k: normalized selected offspring steps with shape (mu, n).
        c1: rank-one learning rate.
        cmu: rank-mu learning rate.
        cc: covariance path accumulation rate.
        hsig: path reliability flag.
        n: problem dimensionality.

    Returns:
        Updated covariance matrix with shape (n, n).
    """
    rank1 = c1 * (np.outer(p_c, p_c) + (1.0 - hsig) * cc * (2.0 - cc) * C)
    rankmu = cmu * np.sum(
        [weights[i] * np.outer(y_k[i], y_k[i]) for i in range(len(weights))],
        axis=0,
    )
    return (1.0 - c1 - cmu) * C + rank1 + rankmu
'''

task_description = (
    "Design a covariance matrix update rule for CMA-ES. The function receives the "
    "current covariance matrix, covariance evolution path, selected normalized "
    "offspring steps, recombination weights, standard CMA-ES learning rates, and "
    "the path reliability flag. The evaluator handles sampling, selection, mean "
    "update, and step-size adaptation. The baseline is the standard rank-one plus "
    "rank-mu covariance update. Performance is measured by final objective value "
    "on Sphere, Rastrigin, Ackley, Rosenbrock, and Griewank benchmarks."
)

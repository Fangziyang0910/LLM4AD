template_program = '''
import numpy as np


def adapt_diagonal_cov(
    d: np.ndarray,
    p_c: np.ndarray,
    weights: np.ndarray,
    y_k: np.ndarray,
    c1: float,
    cmu: float,
    cc: float,
    hsig: float,
    n: int,
    generation: int,
    max_generations: int,
) -> np.ndarray:
    """Return updated sep-CMA-ES diagonal variance factors.

    Args:
        d: current per-dimension variance factors with shape (n,).
        p_c: cumulative covariance evolution path with shape (n,).
        weights: recombination weights for selected offspring.
        y_k: selected normalized offspring steps with shape (mu, n).
        c1: rank-one learning rate.
        cmu: rank-mu learning rate.
        cc: covariance path accumulation rate.
        hsig: path reliability flag.
        n: problem dimensionality.
        generation: current zero-based generation.
        max_generations: total planned generations.

    Returns:
        Updated positive variance factors with shape (n,).
    """
    rank1 = c1 * (p_c ** 2 + (1.0 - hsig) * cc * (2.0 - cc) * d)
    rankmu = cmu * np.einsum("i,ij->j", weights, y_k ** 2)
    return (1.0 - c1 - cmu) * d + rank1 + rankmu
'''

task_description = (
    "Design a diagonal variance adaptation rule for separable CMA-ES on "
    "high-dimensional continuous optimization. The evaluator handles sampling, mean "
    "update, cumulative step-size adaptation, and path updates. The function receives "
    "current diagonal variances, the covariance path, selected normalized offspring "
    "steps, recombination weights, learning rates, a path reliability flag, and "
    "generation progress. The baseline is the standard separable rank-one plus rank-mu "
    "diagonal covariance update. Performance is measured by final objective value on "
    "100-dimensional Sphere, Rastrigin, Ackley, Rosenbrock, and Griewank benchmarks."
)

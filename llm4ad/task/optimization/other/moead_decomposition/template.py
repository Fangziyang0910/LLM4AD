template_program = '''
import numpy as np


def custom_decomposition(F: np.ndarray,
                         weights: np.ndarray,
                         ideal_point: np.ndarray) -> np.ndarray:
    """Aggregate objective vectors into scalar MOEA/D subproblem scores.

    Args:
        F: objective values, shape (n_solutions, n_objectives), lower is better.
        weights: corresponding weight vectors, shape (n_solutions, n_objectives).
        ideal_point: current best objective value per objective, shape (n_objectives,).

    Returns:
        Scalar score per solution with shape (n_solutions,). Lower is better.
    """
    shifted = np.abs(F - ideal_point) * weights
    return np.max(shifted, axis=1)
'''

task_description = (
    "Design a decomposition function for MOEA/D. The evaluator runs a complete "
    "MOEA/D loop on DTLZ2 benchmark instances and uses the function to compare "
    "old neighbor solutions with a child solution for each subproblem. The function "
    "receives objective vectors, weight vectors, and the current ideal point, then "
    "returns scalar scores where lower is better. Performance is measured by final "
    "Pareto-front hypervolume on fixed train/test configurations."
)

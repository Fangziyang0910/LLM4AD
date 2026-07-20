task_description = """
Design a novel decomposition function for Multi-Objective Evolutionary Algorithm
based on Decomposition (MOEA/D). MOEA/D turns a multi-objective problem into
scalar subproblems; your function defines how each subproblem aggregates an
objective vector into one scalar.

The function receives a batch of objective vectors F with shape
(n_solutions, n_objectives), matching weight vectors, and the current ideal
point with shape (n_objectives,). It must return one scalar per solution with
shape (n_solutions,). For each subproblem, a lower scalar means a better
candidate solution (as in Tchebycheff / weighted Chebyshev).

Search performance is measured by the hypervolume of the final Pareto front on
DTLZ; the evaluator returns hypervolume directly so that higher scores are better.
"""

template_program = '''
import numpy as np

def custom_decomposition(F: np.ndarray,
                         weights: np.ndarray,
                         ideal_point: np.ndarray,
                         **kwargs) -> np.ndarray:
    """Design a novel decomposition method for MOEA/D.

    Args:
        F (np.ndarray): Objective vectors, shape (n_solutions, n_objectives).
        weights (np.ndarray): Subproblem weights, shape (n_solutions, n_objectives).
        ideal_point (np.ndarray): Ideal point so far, shape (n_objectives,).

    Returns:
        np.ndarray: Scalar aggregation score per solution, shape (n_solutions,).
            Lower means better for that subproblem.
    """
    # Default implementation: Tchebycheff decomposition.
    v = np.abs(F - ideal_point) * weights
    return np.max(v, axis=1)
'''

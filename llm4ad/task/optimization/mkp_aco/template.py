template_program = '''
import numpy as np


def heuristics(prize: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Return item desirability values for MKP ant colony optimization.

    Args:
        prize: item prize vector with shape (n,).
        weight: normalized item weight matrix with shape (n, m). Each capacity
            constraint is normalized to 1.

    Returns:
        A non-negative vector with shape (n,), where larger values make an item
        more likely to be selected by ants.
    """
    return prize / (np.sum(weight, axis=1) + 1e-9)
'''

task_description = (
    "Given item prizes and a normalized multi-dimensional weight matrix for a "
    "Multiple Knapsack Problem, design item desirability values for ant colony "
    "optimization. Each capacity constraint is normalized to 1. The goal is to "
    "maximize the total prize of selected feasible items."
)

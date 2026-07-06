template_program = '''
import numpy as np


def generate_mutation(current_solution: np.ndarray,
                      sigma: float,
                      success_rate: float,
                      n_dims: int,
                      iteration: int,
                      max_evals: int) -> np.ndarray:
    """Generate the mutation displacement for Nevergrad's OnePlusOne optimizer.

    Args:
        current_solution: Current best solution in standardized space.
        sigma: Nevergrad's current isotropic step size.
        success_rate: Fraction of successful mutations in the recent window.
        n_dims: Problem dimensionality.
        iteration: Current ask counter.
        max_evals: Total evaluation budget.

    Returns:
        Full displacement vector with shape (n_dims,).
    """
    return sigma * np.random.normal(0.0, 1.0, n_dims)
'''

task_description = (
    "Design the mutation noise generator for Nevergrad's OnePlusOne optimizer. "
    "The evaluator plugs the function into a thin subclass of Nevergrad's internal "
    "_OnePlusOne optimizer, replacing only the Gaussian mutation step. Nevergrad's "
    "acceptance logic, current-best bookkeeping, and sigma adaptation remain intact. "
    "Performance is measured by negative mean log1p final best objective across "
    "fixed continuous benchmark functions."
)

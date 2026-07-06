template_program = '''
import numpy as np


def adapt_step_size(
    sigma: float,
    acceptance_rate: float,
    f_parent: float,
    f_offspring: np.ndarray,
    n: int,
    generation: int,
    max_generations: int,
) -> float:
    """Return the next step size for a (1+lambda)-ES.

    Args:
        sigma: current isotropic Gaussian mutation standard deviation.
        acceptance_rate: EMA-smoothed fraction of offspring improving the parent.
        f_parent: current parent fitness, lower is better.
        f_offspring: raw offspring fitness values from this generation.
        n: problem dimensionality.
        generation: current zero-based generation.
        max_generations: total planned generations.

    Returns:
        Positive next sigma. The evaluator clips it to [1e-12, domain_width].
    """
    c = 0.817
    if acceptance_rate > 0.2:
        return sigma / c
    if acceptance_rate < 0.2:
        return sigma * c
    return sigma
'''

task_description = (
    "Design a step-size adaptation rule for a (1+lambda)-Evolution Strategy. "
    "Each generation samples lambda offspring from N(parent, sigma^2 I), accepts the "
    "best improving offspring, then calls the designed function to update sigma. "
    "The function receives current sigma, an EMA-smoothed acceptance rate, parent "
    "fitness, all offspring fitness values, dimensionality, and generation progress. "
    "The baseline is Rechenberg's 1/5-success rule. Performance is measured by final "
    "objective value on Sphere, Rastrigin, Ackley, Rosenbrock, and Griewank benchmarks."
)

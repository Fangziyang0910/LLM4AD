template_program = '''
import numpy as np


def acquisition(mu: np.ndarray, sigma: np.ndarray, f_best: float) -> np.ndarray:
    """Score Bayesian optimization candidates.

    Args:
        mu: GP predictive mean for each candidate point. Lower is better.
        sigma: GP predictive standard deviation for each candidate point.
        f_best: lowest objective value observed so far.

    Returns:
        Acquisition score for each candidate. The BO loop evaluates the
        candidate with the highest score.
    """
    kappa = 2.0
    return -mu + kappa * sigma
'''

task_description = (
    "Design an acquisition function for Bayesian optimization. The acquisition "
    "function receives the Gaussian Process predictive mean and standard "
    "deviation for a batch of candidate points, plus the best observed objective "
    "value so far. It returns a score per candidate; the candidate with the "
    "highest score is evaluated next. The benchmarks are minimization problems, "
    "and performance is measured by simple regret."
)

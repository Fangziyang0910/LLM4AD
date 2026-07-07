template_program = '''
import numpy as np


def crossover(x1: np.ndarray, x2: np.ndarray) -> tuple:
    """Return two offspring from two parent vectors for continuous NSGA-II.

    Args:
        x1: first parent decision vector with values in [0, 1].
        x2: second parent decision vector with values in [0, 1].

    Returns:
        Tuple (c1, c2), both with the same shape as the parents.
    """
    eta = 15.0
    c1, c2 = x1.copy(), x2.copy()
    for i in range(len(x1)):
        if np.random.random() < 0.5 and abs(x1[i] - x2[i]) > 1e-10:
            u = np.random.random()
            beta = (
                (2 * u) ** (1.0 / (eta + 1))
                if u <= 0.5
                else (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1))
            )
            c1[i] = 0.5 * ((x1[i] + x2[i]) - beta * abs(x2[i] - x1[i]))
            c2[i] = 0.5 * ((x1[i] + x2[i]) + beta * abs(x2[i] - x1[i]))
    return c1, c2
'''

task_description = (
    "Design a crossover operator for pymoo's NSGA-II on continuous ZDT problems. "
    "The function receives two parent vectors and returns two offspring vectors. "
    "The evaluator plugs it into pymoo NSGA2 with standard polynomial mutation and "
    "rank-and-crowding survival. Performance is measured by final hypervolume on "
    "fixed train/test ZDT configurations."
)

template_program = '''
import numpy as np

def equation(xs: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Equation skeleton fitted to observed data.

    Called during evaluation to predict outputs from inputs. The evaluator
    optimizes ``params`` to minimize mean squared error against observations.

    Args:
        xs: 2-d array of shape (n_points, n_features). Column i is variable i.
        params: 1-d array of numeric constants optimized by the evaluator.

    Return:
        1-d array of predictions with length n_points.
    """
    return params[0] * xs[:, 0] + params[1] * xs[:, 0] + params[2]
'''

task_description = (
    "Design a mathematical equation skeleton that maps observed input features to a target. "
    "The function receives a 2-d array xs of shape (n_points, n_features) and a parameter "
    "vector params. It must return a 1-d prediction array of length n_points. You do not need "
    "to use every parameter. The evaluator fits params by minimizing mean squared error; "
    "the search score is the negated MSE (higher is better)."
)

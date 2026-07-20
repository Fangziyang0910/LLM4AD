template_program = '''
import numpy as np

def equation(t: np.ndarray, x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Equation skeleton for oscillator acceleration with explicit time.

    Args:
        t: Time observations.
        x: Position observations.
        v: Velocity observations.
        params: Numeric constants fitted by the evaluator.

    Return:
        Predicted acceleration array of the same length.
    """
    dv = params[0] * t + params[1] * x + params[2] * v + params[3]
    return dv
'''

task_description = (
    "Design an equation skeleton for acceleration in a damped nonlinear oscillator with "
    "driving force. The function receives time t, position x, velocity v, and parameter "
    "vector params, and must return predicted acceleration. The evaluator fits params by "
    "MSE; the search score is the negated MSE (higher is better)."
)

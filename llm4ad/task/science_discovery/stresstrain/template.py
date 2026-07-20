template_program = '''
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Equation skeleton for stress in an aluminium rod.

    Args:
        strain: Strain observations.
        temp: Temperature observations.
        params: Numeric constants fitted by the evaluator.

    Return:
        Predicted stress array of the same length.
    """
    return params[0] * strain + params[1] * temp
'''

task_description = (
    "Design an equation skeleton for stress given strain and temperature observations of "
    "an aluminium rod (elastic and plastic regimes). The function receives strain, temp, "
    "and parameter vector params, and must return predicted stress. The evaluator fits "
    "params by MSE; the search score is the negated MSE (higher is better)."
)

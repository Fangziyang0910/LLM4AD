template_program = '''
import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Equation skeleton for bacterial growth rate.

    Called to predict growth rate from observations. The evaluator optimizes
    params by minimizing MSE against measured rates.

    Args:
        b: Population density observations.
        s: Substrate concentration observations.
        temp: Temperature observations.
        pH: pH observations.
        params: Numeric constants fitted by the evaluator.

    Return:
        Predicted growth-rate array with the same length as the inputs.
    """
    return params[0] * b + params[1] * s + params[2] * temp + params[3] * pH + params[4]
'''

task_description = (
    "Design a mathematical equation skeleton for E. coli bacterial growth rate. "
    "The function receives aligned observation arrays for population density b, "
    "substrate s, temperature temp, and pH, plus a parameter vector params. "
    "It must return a predicted growth-rate array of the same length. The evaluator "
    "fits params by MSE; the search score is the negated MSE (higher is better)."
)

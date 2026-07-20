template_program = '''
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """ODE right-hand side dx/dt = f(x; params).

    During evaluation the signature is rewritten to equation(t, x, params) for
    scipy.integrate.solve_ivp. Design f using the state value x and parameters.

    Args:
        x: Length-one array containing the current state value of the ODE.
        params: A 1-d array of numeric constants or parameters to be optimized.

    Return:
        A length-one array giving the derivative dx/dt at the current state.
    """
    y = params[0] * x + params[2]
    return y
'''

task_description = (
    "Find an ODE right-hand-side skeleton dx/dt = f(x; params) that fits observed trajectories "
    "from given initial states. The function should be differentiable and continuous. "
    "Only selectable components: "
    "1. Basic operators: +, -, *, /, **, np.sqrt, np.exp, np.log, np.abs "
    "2. Trigonometric expressions: np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan "
    "3. Standard constants: np.pi for pi and np.e for Euler's number. "
    "Do not use the bitwise/xor operator '^'; use '**' for powers."
)

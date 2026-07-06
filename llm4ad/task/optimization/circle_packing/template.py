template_program = '''
import numpy as np


def construct_packing():
    """Construct 26 circles in a unit square.

    Returns:
        centers: numpy array of shape (26, 2), with x/y coordinates.
        radii: numpy array of shape (26,), with non-negative radii.
    """
    n = 26
    xs = np.linspace(0.1, 0.9, 6)
    ys = np.linspace(0.1, 0.9, 5)
    centers = np.array([[x, y] for y in ys for x in xs], dtype=float)[:n]
    radii = np.full(n, 0.06, dtype=float)
    return centers, radii
'''

task_description = (
    "Construct a packing of 26 circles inside the unit square. The returned "
    "centers must have shape (26, 2), radii must have shape (26,), every circle "
    "must remain fully inside [0, 1]^2, and no two circles may overlap. The "
    "score is the sum of radii, so higher is better."
)

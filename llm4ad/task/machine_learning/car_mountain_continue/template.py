template_program = '''
import numpy as np

def choose_action(pos: float, v: float, last_action: float) -> float:
    """Return the action for the car to proceed the next move.
    Args:
        pos: Car's position, a float ranges between [-1.2, 0.6].
        v: Car's velocity, a float ranges between [-0.07, 0.07].
        last_action: Car's next move, a float ranges between [-1, 1].
    Return:
         A [float] representing the force to be applied to the car.
         The value should be in the range of [-1.0, 1.0].
    """
    return np.random.uniform(-1.0, 1.0)
'''

task_description = ("Implement a continuous-force strategy that drives a car along an uneven road toward a goal. "
                    "At each step apply a force in [-1, 1] based on the car's current position, velocity, and "
                    "previous force. Successful episodes are scored by how quickly the goal is reached; failed "
                    "episodes are scored by how close the car gets to the goal position.")

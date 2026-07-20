template_program = '''
import numpy as np

def choose_action(pos: float, v: float, last_action: int) -> int:
    """Return the action for the car to proceed the next move.
    Args:
        pos: Car's position, a float ranges between [-1.2, 0.6].
        v: Car's velocity, a float ranges between [-0.07, 0.07].
        last_action: Car's next move, a int ranges between [0, 1, 2].
    Return:
         An integer representing the selected action for the car.
         0: accelerate to left
         1: don't accelerate
         2: accelerate to right
    """
    return np.random.randint(3)
'''

task_description = ("Implement a discrete-action strategy that drives a car along an uneven road toward a goal. "
                    "At each step choose an action from the car's current position, velocity, and previous action. "
                    "Successful episodes are scored by how quickly the goal is reached; failed episodes are scored "
                    "by how close the car gets to the goal position.")

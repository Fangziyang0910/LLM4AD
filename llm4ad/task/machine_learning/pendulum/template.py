template_program = '''
import numpy as np

def choose_action(x: float, y: float, av: float, last_action: float) -> float:
    """
    Args:
        x: cos(theta), between [-1, 1]
        y: sin(theta), between [-1, 1]
        av: angular velocity of the pendulum, between [-8.0, 8.0]
        last_action: the last torque applied to the pendulum, a float between [-2.0, 2.0]

    Return:
         A float representing the torque to be applied to the pendulum.
         The value should be in the range of [-2.0, 2.0].
    """
    action = np.random.uniform(-2.0, 2.0)
    return action
'''

task_description = ("Implement a novel control strategy for the inverted pendulum swing-up problem. At each step "
                    "apply a torque based on cos(theta), sin(theta), angular velocity, and the previous torque to "
                    "reach and stabilize the upright position. Fitness is based on the final uprightness/stability "
                    "error at episode end; if the pendulum is perfectly upright, shorter episodes are preferred.")

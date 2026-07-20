template_program = '''
import numpy as np

def choose_action(ct1: float, st1: float, ct2: float, st2: float, avt1: float, avt2: float, last_action: int) -> int: 
    """
    Design a novel algorithm to select the action in each step.

    Args:
        ct1: cosine of theta1, float between [-1, 1].
        st1: sine of theta1, float between [-1, 1]
        ct2: cosine of theta2, float between [-1, 1].
        st2: sine of theta2, float between [-1, 1].
        avt1: angular velocity of theta1, float between [-12.567, 12.567].
        avt2: angular velocity of theta2, float between [-28.274, 28.274].
        last_action: Previous action, one of 0, 1, or 2.

    Return:
         An integer representing the selected action for the acrobot.
         0: apply -1 torque on actuated  joint.
         1: apply 0 torque on actuated joint
         2: apply +1 torque on actuated joint.

    """
    # this is a placehold, replace it with your algorithm
    action =  np.random.randint(3)

    return action
'''

task_description = ("Design an innovative control heuristic for an acrobot. At each step the function selects an "
                    "action from joint angles and angular velocities (and the previous action) to swing the lower "
                    "link and raise the tip to the target height. Fitness rewards reaching the goal in fewer steps; "
                    "if the episode ends without success, it uses a height-based proxy of how close the tip is to "
                    "the goal.")

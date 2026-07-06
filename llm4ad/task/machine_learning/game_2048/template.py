template_program = '''
import numpy as np


def get_best_move(board: np.ndarray) -> int:
    """Choose a move for a 2048 board.

    Args:
        board: A 4x4 array. Empty cells are 0; a tile with value 2**k is stored as k.

    Returns:
        0 for up, 1 for down, 2 for left, or 3 for right.
    """
    return 2
'''

task_description = (
    "Design a fast policy for the 2048 game. The board is a 4x4 numpy array "
    "where empty cells are 0 and tile values are represented by powers of two. "
    "The policy returns one move: 0 up, 1 down, 2 left, or 3 right. The score "
    "rewards reaching high tiles with fewer actions on a fixed-seed game."
)

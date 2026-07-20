template_program = '''
import numpy as np
import scipy.optimize as opt
import math
import random
from typing import List, Tuple, Dict
def solve(nodes: list) -> dict:
    """
    Solve a TSP instance.
    Args:
        - nodes (list): List of (x, y) coordinates representing cities in the TSP problem
                     Format: [(x1, y1), (x2, y2), ..., (xn, yn)]
    Returns:
        dict: Solution information with:
            - 'tour' (list): List of node indices representing the solution path
                            Format: [0, 3, 1, ...] where numbers are indices into the nodes list
    """

    return {
        'tour': [],
    }
'''

task_description = (
    "The Traveling Salesman Problem (TSP) asks for a shortest tour that visits each city "
    "exactly once and returns to the start. The function receives a list of city coordinates "
    "nodes = [(x, y), ...] and must return a dict {'tour': [...]} with a permutation of "
    "0-based city indices forming a Hamiltonian cycle. Distances are Euclidean and computed "
    "by the evaluator from the coordinates. The objective is to minimize tour length; the "
    "evaluator returns the negated mean length (higher is better)."
)

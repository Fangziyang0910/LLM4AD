template_program = '''
import numpy as np
import scipy.optimize as opt
import math
import random
from typing import List, Tuple, Dict
def solve(n: int, edges: list, adjacency: dict) -> dict:
    """
    Problem:
        Given a graph in DIMACS format (with vertices, edges, and an adjacency list),
        assign a positive integer color to each vertex (1..n) so that no two adjacent vertices
        share the same color. The objective is to use as few colors as possible.
    Input arguments:
    The keyword arguments are expected to include at least:
      - n: int (int), the number of vertices.
      - edges: list of (u, v) tuples (tuple of int (int), int (int)) representing edges.
      - adjacency: dict mapping each vertex (1..n) (int) to a set of its adjacent vertices (set of int).
    Evaluation Metric:
        Let  k  be the number of distinct colors used.
        For every edge connecting two vertices with the same color, count one conflict ( C ).
        If  C > 0 , the solution is invalid and receives no score.
        Otherwise, the score is simply  k , with a lower  k  being better.
    Returns:
        A dictionary representing the solution, mapping each vertex_id (1..n) to a positive integer color.
    """
    ## placeholder.
    return {}  # Replace {} with the actual solution dictionary when implemented.
'''

task_description = (
    "Design a graph colouring solver. Given n vertices, an edge list, and an adjacency map, "
    "assign a positive integer color to every vertex so that no adjacent vertices share a color. "
    "The function must return a dict mapping each vertex id (1..n) to its color. "
    "Feasible solutions are scored by the number of distinct colors used: fewer colors are better. "
    "The evaluator returns the negated color count so that the search treats higher scores as better."
)

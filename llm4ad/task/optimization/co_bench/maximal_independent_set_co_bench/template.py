template_program = '''
import numpy as np
import networkx as nx
import scipy.optimize as opt
import math
import random
from typing import List, Tuple, Dict
def solve(graph: nx.Graph) -> dict:
    """
    Solve the Maximum Independent Set problem for a given test case.

    Input:
        graph (nx.Graph): Undirected graph whose nodes are integers.

    Returns:
        dict: A solution dictionary containing:
            - mis_nodes (list): List of node indices in an independent set.
              Larger feasible sets score higher.
    """
    # Placeholder: return an empty independent set.
    solution = {
        'mis_nodes': [],
    }
    return solution
'''

task_description = (
    "Design a solver for the Maximum Independent Set (MIS) problem. "
    "Given an undirected NetworkX graph G = (V, E), return a largest subset S of vertices "
    "such that no two vertices in S are adjacent. The function receives the graph object and "
    "must return a dict with key 'mis_nodes' listing the chosen node ids. The evaluator checks "
    "independence and scores the set by its size (higher is better)."
)

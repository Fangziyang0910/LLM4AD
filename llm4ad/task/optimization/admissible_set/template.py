
template_program = '''
import math
import numpy as np

def priority(el: tuple[int, ...], n: int = 15, w: int = 10) -> float:
    """Score a candidate vector for greedy inclusion in an admissible set.

    Called for each feasible child vector while growing the set.

    Args:
        el: Flattened candidate vector formed by concatenating length-n
            ternary triples. It has exactly w non-zero entries.
        n: Vector length / dimension parameter of the admissible-set instance.
        w: Required number of non-zero entries.

    Returns:
        A scalar priority. Larger values make the candidate more likely to be
        selected next. The evaluator grows a maximum admissible set and scores
        by set size relative to a known optimum (higher is better).
    """
    return 0.
'''

task_description = (
    "Design a priority function used while constructing an admissible set of "
    "ternary vectors. At each step the search proposes candidate vectors that "
    "still satisfy the admissible-set constraints; your function scores one "
    "candidate vector el (with parameters n and w) and must return a float. "
    "Higher priority means the candidate is preferred for inclusion. The "
    "objective is to maximize the final set size."
)

template_program = '''
import numpy as np


def crossover(parents: np.ndarray, n_pop: int) -> np.ndarray:
    """Generate offspring for the DPP genetic algorithm.

    Args:
        parents: Selected parent population with shape (P, n_decap).
        n_pop: Number of offspring to generate.

    Returns:
        Offspring population with shape (n_pop, n_decap).
    """
    n_parents, n_decap = parents.shape
    left_halves = parents[:, :n_decap // 2]
    right_halves = parents[:, n_decap // 2:]
    parent_pairs = np.stack([
        np.random.choice(range(n_parents), 2, replace=False)
        for _ in range(n_pop)
    ])
    return np.concatenate([
        left_halves[parent_pairs[:, 0]],
        right_halves[parent_pairs[:, 1]],
    ], axis=1)
'''

task_description = (
    "Design a crossover operator for a genetic algorithm that solves the "
    "Decap Placement Problem. The function receives selected parent solutions, "
    "where each row is a list of decap locations on a 10x10 PDN, and must return "
    "n_pop offspring. The evaluator repairs duplicate or infeasible locations "
    "with the same validation step as the reference GA, then scores the resulting "
    "population by the best DPP reward found after several GA generations."
)

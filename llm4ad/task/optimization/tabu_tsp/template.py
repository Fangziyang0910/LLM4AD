template_program = '''
import numpy as np


def score_moves(delta_costs: np.ndarray, is_tabu_mask: np.ndarray,
                best_cost: float, current_cost: float, tabu_ages: np.ndarray,
                iteration: int, max_iterations: int) -> np.ndarray:
    """Score all candidate 2-opt moves for Tabu Search on TSP.

    Args:
        delta_costs: change in tour cost for each move, shape (n_moves,).
        is_tabu_mask: True for currently tabu moves, shape (n_moves,).
        best_cost: best tour cost found so far.
        current_cost: current tour cost before this iteration's move.
        tabu_ages: age of tabu moves, or 0 for non-tabu moves.
        iteration: current iteration index.
        max_iterations: total planned Tabu Search iterations.

    Returns:
        Score array with shape (n_moves,). The highest finite score is executed.
    """
    scores = np.full(len(delta_costs), -np.inf)
    non_tabu = ~is_tabu_mask
    scores[non_tabu] = -delta_costs[non_tabu]
    aspiration = is_tabu_mask & (current_cost + delta_costs < best_cost)
    scores[aspiration] = -delta_costs[aspiration] + 1e6
    return scores
'''

task_description = (
    "Design a move-scoring function for Tabu Search on Euclidean TSP. At each "
    "iteration the evaluator provides all candidate 2-opt moves as arrays and "
    "executes the move with the highest finite score. The score can combine move "
    "cost delta, tabu status, tabu age, aspiration criteria, progress adaptation, "
    "and diversification. Performance is measured by lower final tour cost on "
    "fixed train/test TSP instances."
)

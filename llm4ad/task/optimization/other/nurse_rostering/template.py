template_program = '''
import numpy as np


def score_assignment(nurse_idx: int, shift_type: int, day: int,
                     nurse_workload: np.ndarray, nurse_preferences: np.ndarray,
                     consecutive_days: np.ndarray, last_shift_type: np.ndarray,
                     target_workload: float, n_days: int) -> float:
    """Score assigning one nurse to one shift in a greedy roster constructor.

    Args:
        nurse_idx: nurse index being scored.
        shift_type: 0=morning, 1=afternoon, 2=night.
        day: current day index.
        nurse_workload: shifts assigned so far, shape (n_nurses,).
        nurse_preferences: preference matrix, shape (n_nurses, 3), values -1/0/1.
        consecutive_days: consecutive working days up to yesterday.
        last_shift_type: last worked shift type, or -1 if none.
        target_workload: balanced workload target by the current day.
        n_days: total days in the rostering horizon.

    Returns:
        A finite scalar score. Higher scores are assigned first.
    """
    preference = nurse_preferences[nurse_idx, shift_type]
    workload_gap = nurse_workload[nurse_idx] - target_workload
    consecutive_penalty = max(0.0, float(consecutive_days[nurse_idx]) - 4.0)
    night_morning_penalty = 1.0 if (shift_type == 0 and last_shift_type[nurse_idx] == 2) else 0.0
    return float(preference - 0.5 * workload_gap - 2.0 * consecutive_penalty
                 - 5.0 * night_morning_penalty)
'''

task_description = (
    "Design a shift-assignment scoring function for greedy nurse rostering. "
    "The evaluator builds a roster day by day for 8 nurses and three daily shift "
    "types requiring [2, 2, 1] nurses. For each shift it scores every eligible "
    "nurse and assigns the top scorers. The objective balances workload fairness, "
    "shift preferences, consecutive working-day violations, and night-to-morning "
    "violations on fixed train/test instances."
)

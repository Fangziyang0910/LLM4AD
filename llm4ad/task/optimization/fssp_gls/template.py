template_program = '''
import numpy as np


def get_matrix_and_jobs(
        current_sequence: list,
        time_matrix: np.ndarray,
        m: int,
        n: int,
) -> tuple[np.ndarray, list[int]]:
    """Modify processing times and choose jobs for targeted perturbation.

    Args:
        current_sequence: current permutation of job indices.
        time_matrix: n by m processing-time matrix.
        m: number of machines.
        n: number of jobs.

    Returns:
        new_matrix: modified n by m processing-time matrix.
        perturb_jobs: list of 2 to 5 job indices used by targeted local search.
    """
    job_loads = np.sum(time_matrix, axis=1)
    perturb_jobs = np.argsort(job_loads)[-min(3, n):].tolist()
    return time_matrix.copy(), perturb_jobs
'''

task_description = (
    "Given a flow-shop scheduling problem with n jobs and m machines, design a "
    "guided local search perturbation strategy. At each iteration the heuristic "
    "receives the current job sequence and processing-time matrix, returns a "
    "modified processing-time matrix, and selects 2 to 5 jobs for targeted local "
    "search. The goal is to minimize the final makespan."
)

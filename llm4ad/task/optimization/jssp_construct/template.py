template_program = '''
from typing import Sequence, Tuple, TypedDict, TypeAlias

JobId: TypeAlias = int
MachineId: TypeAlias = int
Time: TypeAlias = int
ProcessingTime: TypeAlias = int

Operation: TypeAlias = Tuple[JobId, MachineId, ProcessingTime]  # (job_id, machine_id, proc_time)


class CurrentStatus(TypedDict):
    machine_status: Sequence[int]
    job_status: Sequence[int]


def determine_next_operation(
    current_status: CurrentStatus,
    feasible_operations: Sequence[Operation],
) -> Operation:
    """Choose the next operation to schedule in a constructive JSSP solver.

    Called once per scheduling decision while building a complete schedule.

    Args:
        current_status: Dict with
            - machine_status: current available time of each machine
            - job_status: current available time of each job
        feasible_operations: Candidate operations as
            (job_id, machine_id, processing_time) tuples.

    Returns:
        One operation from feasible_operations. The evaluator removes that
        exact tuple from the remaining candidate list.
    """
    # Baseline: shortest processing time
    return min(feasible_operations, key=lambda op: op[2])
'''

task_description = (
    "Design a constructive heuristic for the Job Shop Scheduling Problem. "
    "At each step the function chooses one feasible operation to schedule next, "
    "given current machine/job available times and the list of candidate operations. "
    "It must return one of the provided (job_id, machine_id, processing_time) tuples. "
    "The evaluator assembles a full schedule and scores it by negated average makespan "
    "(higher is better)."
)

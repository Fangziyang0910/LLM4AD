"""Resume the existing V10.4 batch in free slots, after V10.5 has priority."""

from pathlib import Path
import json

from experiments.infra.launcher import is_session_alive
from experiments.traceaad_v10_5 import launch as scheduler

PRIORITY_MANIFEST = Path(__file__).resolve().parents[1] / 'traceaad_v10_5/results/batch_20260905.json'


def priority_is_served() -> bool:
    if not PRIORITY_MANIFEST.exists():
        return False
    plan = json.loads(PRIORITY_MANIFEST.read_text())['plan']
    return len(plan) == 15 and all(
        row['status'] in {'finished', 'blocked', 'stopped'} or is_session_alive(row['session'])
        for row in plan
    )


def main() -> None:
    # Reuse the tested capacity/resume scheduler in this process only. This does
    # not alter the separately running V10.5 scheduler or either search method.
    scheduler.RESULTS_ROOT = Path(__file__).resolve().parent / 'results'
    scheduler.MODULE = 'experiments.traceaad_v10_4.run'
    available_slots = scheduler.free_slots
    scheduler.free_slots = lambda: available_slots() if priority_is_served() else {}
    scheduler.main()


if __name__ == '__main__':
    main()

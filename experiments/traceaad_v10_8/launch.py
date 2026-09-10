"""Launch 15 V10.8 runs, balanced 4/4/4/3 across server1/server3/server3b/local."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
from pathlib import Path
import time

from experiments.infra.base import BACKEND_CAPACITY, LaunchItem, TASKS, TASK_SHORT, free_slots, item_is_running, launch_items
from experiments.infra.launcher import check_backends, get_summary_status
from experiments.infra.base import BACKENDS as BACKEND_PROFILES
from experiments.infra.equivalent_backends import prepare_resume
from llm4ad.method.traceaad_v10_5.traceaad import atomic_json

BACKENDS = ('server1', 'server3', 'server3b', 'local')
RESULTS_ROOT = Path(__file__).resolve().parent / 'results'
MODULE = 'experiments.traceaad_v10_8.run'


def build_plan(batch: str, session_prefix: str, allocation_study=False) -> list[dict]:
    if allocation_study:
        # Equivalent model services: assign unstarted runs to available slots.
        return [
            {'task': 'tsp_construct', 'repeat': seed + 1, 'seed': seed, 'backend': None,
             'allocation_arm': arm,
             'run_name': f'{batch}_tsp_{arm}_v108_rep{seed+1}',
             'session': f'{session_prefix}_{arm}_r{seed+1}',
             'attempts': 0, 'status': 'queued'}
            for seed in range(3)
            for arm in ('A', 'B', 'C', 'D')
        ]
    return [
        {'task': task, 'repeat': repeat, 'seed': repeat - 1, 'backend': BACKENDS[(task_index + repeat - 1) % len(BACKENDS)],
         'run_name': f'{batch}_{TASK_SHORT[task]}_v108_rep{repeat}',
         'session': f'{session_prefix}_{TASK_SHORT[task]}_r{repeat}',
         'attempts': 0, 'status': 'queued'}
        for repeat in range(1, 4) for task_index, task in enumerate(TASKS)
    ]


def launch_item(row: dict) -> LaunchItem:
    extra = ('--allocation-arm', row.get('allocation_arm', 'A'))
    return LaunchItem(task=row['task'], repeat=row['repeat'], seed=row['seed'],
                      backend=row['backend'], session=row['session'], run_name=row['run_name'],
                      run_dir=RESULTS_ROOT / row['task'] / row['run_name'], module=MODULE,
                      extra_args=extra)


def allocate(plan: list[dict], available: dict[str, int]) -> list[tuple[dict, str]]:
    """Reserve free slots across equivalent services, including resumed runs."""
    remaining = dict(available)
    assignments = []
    used_by_task = {task: {r['backend'] for r in plan if r['task'] == task and r['backend']}
                    for task in TASKS}
    for row in plan:
        if row['status'] != 'queued':
            continue
        candidates = [b for b in BACKENDS if remaining.get(b, 0) > 0]
        if not candidates:
            continue
        used = used_by_task[row['task']]
        candidates.sort(key=lambda b: (b in used,
                                       (BACKEND_CAPACITY[b] - remaining[b] + 1) / BACKEND_CAPACITY[b],
                                       BACKENDS.index(b)))
        backend = candidates[0]
        remaining[backend] -= 1
        used.add(backend)
        assignments.append((row, backend))
    return assignments


def healthy_slots(available: dict[str, int], checked: set[str]) -> dict[str, int]:
    available = dict(available)
    for backend in BACKENDS:
        if available.get(backend, 0) > 0 and backend not in checked:
            try:
                check_backends([backend])
            except Exception as exc:
                available[backend] = 0
                print(f'{backend} unavailable: {type(exc).__name__}; retry next poll', flush=True)
            else:
                checked.add(backend)
    return available


def refresh(plan: list[dict], max_attempts: int) -> None:
    for row in plan:
        item = launch_item(row)
        if item_is_running(item):
            row['status'] = 'running'
            continue
        status = get_summary_status(item.run_dir)
        if status == 'finished':
            row['status'] = 'finished'
        elif status == 'blocked':
            row['status'] = 'blocked'
        elif row['attempts'] >= max_attempts:
            row['status'] = 'stopped'
        else:
            row['status'] = 'queued'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', default=datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--session-prefix', default='v108')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--max-attempts', type=int, default=5)
    parser.add_argument('--allocation-study', action='store_true', help='TSP, three paired seeds, four allocation arms')
    args = parser.parse_args()
    if args.interval < 1 or args.max_attempts < 1:
        parser.error('interval and max-attempts must be positive')
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = RESULTS_ROOT / f'batch_{args.batch}.json'
    with manifest.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            payload = json.loads(manifest.read_text())
            if payload['session_prefix'] != args.session_prefix:
                raise ValueError('existing batch has a different session prefix')
            if payload.get('method') != 'v108':
                raise ValueError('existing batch is not V10.8')
            if payload.get('allocation_study', False) != args.allocation_study:
                raise ValueError('existing batch has a different experiment design')
        else:
            payload = {'method': 'v108', 'batch': args.batch, 'session_prefix': args.session_prefix,
                       'created_at': datetime.now().astimezone().isoformat(),
                       'allocation_study': args.allocation_study,
                       'plan': build_plan(args.batch, args.session_prefix, args.allocation_study)}
            collisions = [r['session'] for r in payload['plan'] if item_is_running(launch_item(r))]
            if collisions:
                raise RuntimeError(f'session prefix is already in use: {collisions}')
            existing = [str(launch_item(r).run_dir) for r in payload['plan'] if launch_item(r).run_dir.exists()]
            if existing:
                raise RuntimeError(f'run directories exist without this batch manifest: {existing}')
        plan = payload['plan']
        checked = set()
        last_status = None
        while True:
            refresh(plan, args.max_attempts)
            available = free_slots()
            if not args.dry_run:
                available = healthy_slots(available, checked)
            assignments = allocate(plan, available)
            for row, backend in assignments:
                row['backend'] = backend
                if args.dry_run:
                    launch_items([launch_item(row)], dry_run=True)
                    continue
                row['attempts'] += 1
                row['status'] = 'launching'
                atomic_json(manifest, payload)
                prepare_resume(launch_item(row), BACKEND_PROFILES)
                launch_items([launch_item(row)], dry_run=False)
                row['status'] = 'running'
                atomic_json(manifest, payload)
            if args.dry_run:
                print(f'launchable={len(assignments)} queued={len(plan)-len(assignments)} free={available}')
                return
            payload['updated_at'] = datetime.now().astimezone().isoformat()
            atomic_json(manifest, payload)
            counts = dict(Counter(r['status'] for r in plan))
            if counts != last_status:
                print(f'{payload["updated_at"]} {counts} manifest={manifest}', flush=True)
                last_status = counts
            if not args.watch or all(r['status'] in ['finished', 'blocked', 'stopped'] for r in plan):
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()

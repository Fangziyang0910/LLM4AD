"""Queue 15 repaired V10.7 runs using task-specific evidence by default."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import json
from pathlib import Path
import time

from experiments.infra.base import (
    LaunchItem, TASKS, TASK_SHORT, free_slots, item_is_running, launch_items,
)
from experiments.infra.launcher import get_summary_status
from experiments.traceaad_v10_6.launch import allocate, healthy_slots
from llm4ad.method.traceaad_v10_5.traceaad import atomic_json
from llm4ad.method.traceaad_v10_7.sampling import CONTEXT_POLICY, STRUCTURE_PREFERENCE

RESULTS_ROOT = Path(__file__).resolve().parent / 'results'
MODULE = 'experiments.traceaad_v10_7.run'


def build_plan(batch: str, session_prefix: str, max_context_programs=2) -> list[dict]:
    return [
        {
            'task': task, 'repeat': repeat, 'seed': repeat - 1, 'backend': None,
            'run_name': f'{batch}_{TASK_SHORT[task]}_v107r_rep{repeat}',
            'session': f'{session_prefix}_{TASK_SHORT[task]}_r{repeat}',
            'attempts': 0, 'status': 'queued',
            'context_policy': CONTEXT_POLICY, 'max_context_programs': max_context_programs,
            'structure_preference': STRUCTURE_PREFERENCE,
        }
        for repeat in range(1, 4) for task in TASKS
    ]


def launch_item(row: dict) -> LaunchItem:
    return LaunchItem(
        task=row['task'], repeat=row['repeat'], seed=row['seed'],
        backend=row['backend'], session=row['session'], run_name=row['run_name'],
        run_dir=RESULTS_ROOT / row['task'] / row['run_name'], module=MODULE,
        extra_args=('--max-context-programs', str(row['max_context_programs'])),
    )


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
    parser.add_argument('--session-prefix', default='v107r')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--max-attempts', type=int, default=5)
    parser.add_argument('--max-context-programs', type=int, default=2)
    args = parser.parse_args()
    if args.interval < 1 or args.max_attempts < 1 or not 1 <= args.max_context_programs <= 2:
        parser.error('interval and max-attempts must be positive; max-context-programs must be 1 or 2')
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = RESULTS_ROOT / f'batch_{args.batch}.json'
    with manifest.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            payload = json.loads(manifest.read_text())
            if payload['session_prefix'] != args.session_prefix:
                raise ValueError('existing batch has a different session prefix')
            if any(row.get('context_policy', CONTEXT_POLICY) != CONTEXT_POLICY
                   for row in payload['plan']):
                raise ValueError(
                    'existing batch uses a retired context policy; '
                    'resume it with the pinned launch-time code instead'
                )
            if any(row.get('structure_preference') != STRUCTURE_PREFERENCE
                   for row in payload['plan']):
                raise ValueError(
                    'existing batch uses a different structure preference; '
                    'resume it with the pinned launch-time code instead'
                )
            if any(row.get('max_context_programs') != args.max_context_programs
                   for row in payload['plan']):
                raise ValueError('existing batch has a different context configuration')
        else:
            payload = {
                'batch': args.batch, 'session_prefix': args.session_prefix,
                'created_at': datetime.now().astimezone().isoformat(),
                'plan': build_plan(args.batch, args.session_prefix,
                                   args.max_context_programs),
            }
            collisions = [
                row['session'] for row in payload['plan']
                if item_is_running(launch_item(row))
            ]
            if collisions:
                raise RuntimeError(f'session prefix is already in use: {collisions}')
            existing = [
                str(launch_item(row).run_dir) for row in payload['plan']
                if launch_item(row).run_dir.exists()
            ]
            if existing:
                raise RuntimeError(
                    f'run directories exist without this batch manifest: {existing}'
                )
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
                launch_items([launch_item(row)], dry_run=False)
                row['status'] = 'running'
                atomic_json(manifest, payload)
            if args.dry_run:
                print(
                    f'launchable={len(assignments)} queued={len(plan)-len(assignments)} '
                    f'free={available}'
                )
                return
            payload['updated_at'] = datetime.now().astimezone().isoformat()
            atomic_json(manifest, payload)
            counts = dict(Counter(row['status'] for row in plan))
            if counts != last_status:
                print(f'{payload["updated_at"]} {counts} manifest={manifest}', flush=True)
                last_status = counts
            if not args.watch or all(
                row['status'] in ['finished', 'blocked', 'stopped'] for row in plan
            ):
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()

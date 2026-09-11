"""Queue five tasks x three seeds; start at most one run per poll on a free service slot."""

import argparse
from collections import Counter
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time

from experiments.infra.equivalent_backends import prepare_resume
from experiments.infra.base import BACKENDS, LaunchItem, TASKS, TASK_SHORT, free_slots, item_is_running, launch_items
from experiments.infra.launcher import get_summary_status
from experiments.traceaad_v10_8.launch import allocate, healthy_slots
from llm4ad.method.traceaad_v10_5.traceaad import atomic_json

RESULTS_ROOT = Path(__file__).resolve().parent / 'results'


def build_plan(batch, prefix):
    return [dict(task=task, repeat=repeat, seed=repeat-1, backend=None,
                 run_name=f'{batch}_{TASK_SHORT[task]}_v1010_rep{repeat}',
                 session=f'{prefix}_{TASK_SHORT[task]}_r{repeat}', attempts=0, status='queued')
            for repeat in range(1, 4) for task in TASKS]


def launch_item(row):
    return LaunchItem(task=row['task'], repeat=row['repeat'], seed=row['seed'],
                      backend=row['backend'], session=row['session'], run_name=row['run_name'],
                      run_dir=RESULTS_ROOT / row['task'] / row['run_name'],
                      module='experiments.traceaad_v10_10.run')


def refresh(plan, max_attempts):
    for row in plan:
        item = launch_item(row)
        status = get_summary_status(item.run_dir)
        if item_is_running(item):
            row['status'] = 'running'
        elif status in ('finished', 'blocked'):
            row['status'] = status
        elif row['attempts'] >= max_attempts:
            row['status'] = 'stopped'
        else:
            row['status'] = 'queued'


def verify_runtime():
    root = Path(__file__).resolve().parents[2]
    manifest = root / 'runtime_manifest.json'
    if not manifest.exists():
        raise ValueError('freeze the reviewed source first using experiments.traceaad_v10_10.freeze')
    payload = json.loads(manifest.read_text())
    for relative, expected in payload['files'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'frozen source changed: {relative}')
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--session-prefix', default='v1010')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--max-attempts', type=int, default=3)
    args = parser.parse_args()
    if args.interval < 1 or args.max_attempts < 1:
        parser.error('interval and max-attempts must be positive')
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in (args.batch, args.session_prefix)):
        parser.error('batch and prefix must contain only letters, numbers, underscore or hyphen')
    identity = None if args.dry_run else verify_runtime()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = RESULTS_ROOT / f'batch_{args.batch}.json'
    with manifest.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            payload = json.loads(manifest.read_text())
            if (payload['method'], payload['session_prefix']) != ('v1010', args.session_prefix):
                raise ValueError('batch identity mismatch')
            if not args.dry_run and payload['source_identity'] != identity:
                raise ValueError('batch frozen source mismatch')
        else:
            payload = dict(method='v1010', batch=args.batch, session_prefix=args.session_prefix,
                           source_identity=identity, created_at=datetime.now().astimezone().isoformat(),
                           plan=build_plan(args.batch, args.session_prefix))
            if any(item_is_running(launch_item(r)) or launch_item(r).run_dir.exists()
                   for r in payload['plan']):
                raise ValueError('existing session or run directory without matching batch manifest')
        last = None
        while True:
            plan = payload['plan']
            refresh(plan, args.max_attempts)
            available = free_slots()
            if not args.dry_run:
                # Recheck service health on every actual start; no stale health cache.
                available = healthy_slots(available, set())
            assignments = allocate(plan, available)[:1]
            if args.dry_run:
                print(json.dumps(dict(plan=plan, free=available, next=[b for _, b in assignments]), indent=2))
                return
            for row, backend in assignments:
                # A competing legacy watcher may have occupied the slot during health checks.
                if free_slots().get(backend, 0) <= 0:
                    continue
                row.update(backend=backend, attempts=row['attempts']+1, status='launching')
                atomic_json(manifest, payload)
                try:
                    prepare_resume(launch_item(row), BACKENDS)
                    launch_items([launch_item(row)], dry_run=False)
                except Exception as exc:
                    row.update(status='queued', last_launch_error=str(exc))
                else:
                    row['status'] = 'running'
            payload['updated_at'] = datetime.now().astimezone().isoformat()
            atomic_json(manifest, payload)
            counts = dict(Counter(r['status'] for r in plan))
            if counts != last:
                print(f'{payload["updated_at"]} {counts} manifest={manifest}', flush=True)
                last = counts
            if not args.watch or all(r['status'] in ('finished', 'blocked', 'stopped') for r in plan):
                return
            time.sleep(args.interval)


if __name__ == '__main__':
    main()

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

from experiments.infra.base import BACKEND_CAPACITY, BACKENDS, LaunchItem, TASKS, TASK_SHORT, free_slots, item_is_running, launch_items
from experiments.infra.launcher import check_backends, get_summary_status
from experiments.infra.equivalent_backends import prepare_resume
from llm4ad.method.traceaad_v10_11.core import atomic_json

RESULTS_ROOT = Path(__file__).resolve().parent / 'results'
BACKEND_NAMES = tuple(BACKENDS)


def allocate(plan, available, backend_pool):
    remaining, assignments = dict(available), []
    used_by_task = {task: {row['backend'] for row in plan
                           if row['task'] == task and row['backend']}
                    for task in TASKS}
    for row in plan:
        if row['status'] != 'queued':
            continue
        candidates = [backend for backend in backend_pool if remaining.get(backend, 0) > 0]
        if not candidates:
            continue
        used = used_by_task[row['task']]
        candidates.sort(key=lambda backend: (
            backend in used,
            (BACKEND_CAPACITY[backend] - remaining[backend] + 1) / BACKEND_CAPACITY[backend],
            BACKEND_NAMES.index(backend)))
        backend = candidates[0]
        remaining[backend] -= 1
        used.add(backend)
        assignments.append((row, backend))
    return assignments


def healthy_slots(available, backend_pool):
    available = dict(available)
    for backend in backend_pool:
        if available.get(backend, 0) > 0:
            try:
                check_backends([backend])
            except Exception:
                available[backend] = 0
    return available


def build_plan(batch, prefix, thinking=False, history_code=False):
    return [dict(task=task, repeat=repeat, seed=repeat-1, backend=None,
                 run_name=f'{batch}_{TASK_SHORT[task]}_v1011_rep{repeat}',
                 session=f'{prefix}_{TASK_SHORT[task]}_r{repeat}', attempts=0, status='queued',
                 **({'thinking': True} if thinking else {}),
                 **({'history_code': True} if history_code else {}))
            for repeat in range(1, 4) for task in TASKS]


def launch_item(row):
    return LaunchItem(task=row['task'], repeat=row['repeat'], seed=row['seed'],
                      backend=row['backend'], session=row['session'], run_name=row['run_name'],
                      run_dir=RESULTS_ROOT / row['task'] / row['run_name'],
                      module='experiments.traceaad_v10_11.run',
                      extra_args=tuple(arg for arg, enabled in (
                          ('--thinking', row.get('thinking')),
                          ('--history-code', row.get('history_code')),
                      ) if enabled))


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
        raise ValueError('freeze the reviewed source first using experiments.traceaad_v10_11.freeze')
    payload = json.loads(manifest.read_text())
    for relative, expected in payload['files'].items():
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'frozen source changed: {relative}')
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--session-prefix', default='v1011')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--backends', default=','.join(BACKEND_NAMES),
                        help='comma-separated backend pool for this batch')
    parser.add_argument('--direct', action='store_true',
                        help='start all currently available assignments in each poll')
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--thinking', action='store_true',
                        help='stamp every run of this batch with model thinking mode')
    parser.add_argument('--history-code', action='store_true',
                        help='include historical programs in each formation path')
    args = parser.parse_args()
    if args.interval < 1 or args.max_attempts < 1:
        parser.error('interval and max-attempts must be positive')
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', s) for s in (args.batch, args.session_prefix)):
        parser.error('batch and prefix must contain only letters, numbers, underscore or hyphen')
    backend_pool = tuple(dict.fromkeys(args.backends.split(',')))
    if not backend_pool or any(backend not in BACKENDS for backend in backend_pool):
        parser.error(f'backends must be drawn from {", ".join(BACKEND_NAMES)}')
    identity = None if args.dry_run else verify_runtime()
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = RESULTS_ROOT / f'batch_{args.batch}.json'
    with manifest.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            payload = json.loads(manifest.read_text())
            if (payload['method'], payload['session_prefix']) != ('v1011', args.session_prefix):
                raise ValueError('batch identity mismatch')
            if tuple(payload.get('backends', ())) != backend_pool:
                raise ValueError('batch backend pool mismatch')
            if bool(payload.get('history_code')) != args.history_code:
                raise ValueError('batch history-code mismatch')
            if not args.dry_run and payload['source_identity'] != identity:
                raise ValueError('batch frozen source mismatch')
        else:
            payload = dict(method='v1011', batch=args.batch, session_prefix=args.session_prefix,
                           source_identity=identity, created_at=datetime.now().astimezone().isoformat(),
                           thinking=args.thinking, history_code=args.history_code,
                           backends=backend_pool, direct=args.direct,
                           plan=build_plan(args.batch, args.session_prefix, args.thinking,
                                           args.history_code))
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
                available = healthy_slots(available, backend_pool)
            assignments = allocate(plan, available, backend_pool)
            if not args.direct:
                assignments = assignments[:1]
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

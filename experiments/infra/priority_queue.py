"""Resume a frozen lower-priority batch after all prerequisite runs finish."""

import argparse
import json
import os
from pathlib import Path
import time


def dependencies_finished(paths):
    """Missing, blocked and stopped prerequisites must not release the queue."""
    for path in paths:
        if not path.is_file():
            return False
        data = json.loads(path.read_text())
        plan = data.get('plan', [])
        if not plan or any(row.get('status') != 'finished' for row in plan):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--after-batch', type=Path, action='append', required=True)
    parser.add_argument('--interval', type=int, default=30)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('interval must be positive')
    source = json.loads(args.source_manifest.read_text())
    command = source['launch_command']
    runtime = Path(source['runtime'])
    if not runtime.is_dir() or not command or not Path(command[0]).is_file():
        raise ValueError('frozen runtime or interpreter is missing')
    print('Waiting for priority batches: ' + ', '.join(str(p) for p in args.after_batch), flush=True)
    while not dependencies_finished(args.after_batch):
        time.sleep(args.interval)
    print('Priority batches finished; resuming the original frozen batch.', flush=True)
    os.chdir(runtime)
    # Replace this watcher, retaining the same tmux identity and original batch lock.
    os.execv(command[0], command)


if __name__ == '__main__':
    main()

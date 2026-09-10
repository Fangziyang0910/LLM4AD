"""Use any free equivalent service while preserving frozen search code and progress.

Run this file directly with --source-manifest to adapt an existing frozen launcher.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.routing.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def prepare_resume(item, profiles):
    """Only change service routing metadata; candidate, RNG and receipts stay intact."""
    state_path = item.run_dir / 'tree_state.json'
    if not state_path.exists():
        return
    profile = profiles[item.backend]
    state = json.loads(state_path.read_text())
    llm = state['mechanism']['llm']
    route = {'base_url': profile.base_url, 'model': profile.model}
    if any(llm.get(k) != v for k, v in route.items()):
        backup = item.run_dir / 'checkpoint_before_service_routing.json'
        if not backup.exists():
            write_json(backup, state)
        llm.update(route)
        write_json(state_path, state)
    config_path = item.run_dir / 'run_config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
        config['backend'] = item.backend
        config['llm'].update(route)
        if 'no_proxy' in config['llm']:
            config['llm']['no_proxy'] = profile.no_proxy
        write_json(config_path, config)


def allocate_anywhere(allocate, plan, available):
    # The frozen allocator still expects a pinned backend. Clear only the
    # scheduling copy; return the original rows for its normal persistence.
    by_name = {row['run_name']: row for row in plan}
    candidates = [{**row, 'backend': None} if row['status'] == 'queued' else row for row in plan]
    return [(by_name[row['run_name']], backend) for row, backend in allocate(candidates, available)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=10)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('interval must be positive')
    source = json.loads(args.source_manifest.read_text())
    runtime = Path(source['runtime'])
    for relative, expected in source['files'].items():
        if hashlib.sha256((runtime / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'frozen source changed: {relative}')
    # Direct script execution keeps experiments unimported until the frozen
    # runtime is first on the path. Child runs retain its original cwd/source.
    sys.path.insert(0, str(runtime))
    command = source['launch_command']
    launcher = importlib.import_module(command[2])
    if not Path(launcher.__file__).resolve().is_relative_to(runtime.resolve()):
        raise ValueError('launcher was not imported from the frozen runtime')
    base = importlib.import_module('experiments.infra.base')
    allocate, launch_items = launcher.allocate, launcher.launch_items
    launcher.allocate = lambda plan, available: allocate_anywhere(allocate, plan, available)

    def start(items, *, dry_run):
        for item in items:
            if not dry_run:
                if base.item_is_running(item):
                    raise RuntimeError(f'run is already active: {item.run_name}')
                prepare_resume(item, base.BACKENDS)
            launch_items([item], dry_run=dry_run)

    launcher.launch_items = start
    sys.argv = [command[2], *command[3:], '--interval', str(args.interval)]
    launcher.main()


if __name__ == '__main__':
    main()

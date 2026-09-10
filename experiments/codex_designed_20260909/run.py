"""Evaluate five directly authored programs; no external generation calls."""
import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

from experiments.infra.base import build_task, TASKS
from llm4ad.base.evaluate import SecureEvaluator

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=['train', 'heldout_panel', 'fresh_panel'], default='train')
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--tag', default='v1')
    parser.add_argument('--program-tag', help='read archived task_TAG.py rather than current programs/')
    args = parser.parse_args()
    out = ROOT / f'results_{args.tag}_{args.split}.jsonl'
    for task in args.tasks:
        source = ROOT / f'{task}_{args.program_tag}.py' if args.program_tag else ROOT / 'programs' / f'{task}.py'
        program = source.read_text()
        evaluation, config = build_task(task, 4)
        if args.split != 'train':
            if task in ('cvrp_aco', 'op_aco'):
                config['split'] = 'val_50' if args.split == 'fresh_panel' else 'test_50'
                evaluation = type(evaluation)(**config)
                count = 10 if task == 'cvrp_aco' else 5
                evaluation._datasets = evaluation._datasets[:count]
                config['panel'] = f"first {count} instances of {config['split']}; not full split"
            else:
                config['seed'] = 2026 if args.split == 'fresh_panel' else 2025
                evaluation = type(evaluation)(**{k: v for k, v in config.items() if k != 'split'})
                config['split'] = f"same_scale_seed_{config['seed']}"
        print(f'START {task} {args.split}', flush=True)
        started = time.monotonic()
        outcome = SecureEvaluator(evaluation).evaluate_program_with_details(program)
        row = {'task': task, 'tag': args.tag, 'split': args.split,
               'score': outcome.result, 'failure': outcome.failure_kind,
               'seconds': time.monotonic() - started, 'config': config,
               'program_sha256': hashlib.sha256(program.encode()).hexdigest(),
               'timestamp': datetime.now().astimezone().isoformat()}
        with out.open('a') as f:
            f.write(json.dumps(row) + '\n')
        (ROOT / f'{task}_{args.tag}.py').write_text(program)
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()

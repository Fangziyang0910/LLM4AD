"""Read-only V10.x diagnostics; never generates candidates or calls an evaluator.

Run from the repository root with .venv/bin/python. Optional --tokenize uses
the configured model server's tokenizer on a deterministic response sample.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[3]
TASKS = ['tsp_construct', 'cvrp_aco', 'op_aco', 'online_bin_packing', 'vrptw_construct']


def summary(values):
    return {'n': len(values), 'mean': statistics.mean(values) if values else None,
            'std': statistics.stdev(values) if len(values) > 1 else None,
            'median': statistics.median(values) if values else None}


def code_key(code):
    try:
        return ast.dump(ast.parse(code), include_attributes=False)
    except SyntaxError:
        return code


def analyze_run(path, version):
    raw = path.read_bytes()
    state = json.loads(raw)
    nodes = state['nodes']
    by_id = {n['id']: n for n in nodes}
    best = max(nodes, key=lambda n: n['fitness'])
    result = {'path': str(path.parent.relative_to(ROOT)),
              'state_sha256': hashlib.sha256(raw).hexdigest(),
              'budget_used': state['budget_used'], 'n_nodes': len(nodes),
              'best': best['fitness'], 'best_id': best['id'],
              'root_best': max(n['fitness'] for n in nodes if n['parent_id'] is None),
              'idea_chars': summary([len(n['idea']) for n in nodes]),
              'code_chars': summary([len(n['code']) for n in nodes]),
              'ast_unique_count': len({code_key(n['code']) for n in nodes})}
    if version == 1:
        return result, []
    valid = sorted([n for n in nodes if n['evaluation_id'] <= 400], key=lambda n: n['evaluation_id'])
    result['best_at_400'] = max(n['fitness'] for n in valid)
    result['operators_400'] = {}
    frontier = -math.inf
    for n in valid:
        op = n['operator']
        if n['parent_id'] is not None:
            item = result['operators_400'].setdefault(op, Counter())
            item['valid'] += 1
            q = n['fitness']
            parent_q = by_id[n['parent_id']]['fitness']
            donor_q = by_id[n['donor_id']]['fitness'] if n['donor_id'] is not None else -math.inf
            item['parent_improved'] += int(q > parent_q)
            item['both_improved'] += int(q > max(parent_q, donor_q))
            item['global_improved'] += int(q > frontier)
            item['global_gain'] += max(q - frontier, 0)
        frontier = max(frontier, n['fitness'])
    consumed = 0
    attempts = Counter()
    statuses = Counter()
    event_lines = 0
    with (path.parent / 'events.jsonl').open() as handle:
        for line in handle:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if consumed >= 400:
                break
            event_lines += 1
            attempts[e['operator']] += 1
            statuses[e['status']] += 1
            if e['status'] in ['ok', 'eval_failed']:
                consumed += 1
    result['attempts_through_400_evals'] = dict(attempts)
    result['statuses_through_400_evals'] = dict(statuses)
    result['event_prefix_lines'] = event_lines
    if version != 4:
        return result, []
    ideas, code_calls = [], []
    with (path.parent / 'llm_calls.jsonl').open() as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                call = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Text audit window is generation steps 0..399, not evaluator slots.
            if call['step'] >= 400 or not call.get('response'):
                continue
            if call['stage'] == 'code':
                code_calls.append(call['seconds'])
                continue
            prompt = call['prompt']
            start = prompt.find('# Recent Algorithm Improvement History')
            end = prompt.find('# Reference Algorithm', start)
            if end < 0:
                end = prompt.find('# Design Direction', start)
            history_chars = end - start if start >= 0 and end > start else 0
            ideas.append({'line': line_no, 'step': call['step'], 'operator': call['operator'],
                          'ts': call['ts'], 'parent_id': call['parent_id'],
                          'prompt_chars': len(prompt), 'history_chars': history_chars,
                          'response': call['response'], 'seconds': call['seconds']})
    result['idea_audit_window'] = 'LLM generation steps 0..399; distinct from 400-eval performance prefix'
    result['idea_calls_in_window'] = len(ideas)
    result['idea_response_chars'] = summary([len(x['response']) for x in ideas])
    result['idea_prompt_chars'] = summary([x['prompt_chars'] for x in ideas])
    result['history_chars'] = summary([x['history_chars'] for x in ideas if x['parent_id'] is not None])
    result['idea_seconds'] = summary([x['seconds'] for x in ideas])
    result['code_seconds'] = summary(code_calls)
    roots = [x for x in ideas if x['parent_id'] is None]
    extensions = [x for x in ideas if x['parent_id'] is not None]
    # Two first Init calls and four equally spaced expansion calls per run.
    chosen = roots[:2]
    if extensions:
        chosen += [extensions[round((len(extensions) - 1) * j / 3)] for j in range(4)]
    config = json.loads((path.parent / 'run_config.json').read_text())['llm']
    samples = [{'run_path': result['path'], 'config': config, **x} for x in chosen]
    return result, samples


def tokenize_sample(sample):
    from experiments.infra.base import build_llm_client
    config = sample['config']
    client = build_llm_client(base_url=config['base_url'], model=config['model'],
                              no_proxy=config['no_proxy'], max_tokens=1024)
    out = {k: v for k, v in sample.items() if k not in ['config', 'response']}
    out['response_chars'] = len(sample['response'])
    out['tail'] = sample['response'][-220:]
    try:
        out['retokenized_tokens'] = client.count_tokens(sample['response'])
    except Exception as error:
        out['tokenizer_error'] = type(error).__name__
    finally:
        client.close()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokenize', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('snapshot.json'))
    args = parser.parse_args()
    output = {'captured_at': datetime.now().astimezone().isoformat(),
              'method': 'State files are read once per run; live runs may advance while reading. Only 20260904b is used for V10.4.',
              'runs': {}, 'aggregates': {}, 'idea_token_samples': []}
    samples = []
    for version in range(1, 5):
        key = f'v10.{version}'
        output['runs'][key] = {}
        output['aggregates'][key] = {}
        for task in TASKS:
            base = ROOT / f'experiments/traceaad_v10_{version}/results' / task
            pattern = ('20260904b*' if version == 4 else '2026*') + '/tree_state.json'
            rows = []
            for path in sorted(base.glob(pattern)):
                row, selected = analyze_run(path, version)
                rows.append(row)
                samples.extend(selected)
            output['runs'][key][task] = rows
            output['aggregates'][key][task] = {
                'complete_1000': summary([r['best'] for r in rows if r['budget_used'] == 1000]),
                'first_400': summary([r['best_at_400'] for r in rows if r['budget_used'] >= 400 and 'best_at_400' in r]),
                'root_best': summary([r['root_best'] for r in rows])}
    if args.tokenize:
        with ThreadPoolExecutor(max_workers=3) as pool:
            output['idea_token_samples'] = list(pool.map(tokenize_sample, samples))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {args.output}; {len(samples)} deterministic Idea samples, {len(output["idea_token_samples"])} tokenized.')


if __name__ == '__main__':
    main()

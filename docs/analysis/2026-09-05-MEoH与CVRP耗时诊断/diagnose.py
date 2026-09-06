"""Read-only run snapshot and bounded prior-function microbenchmark.

Never invokes ACO, SecureEvaluator, or the LLM. Microbenchmarks are auxiliary
diagnostics, separate from the formal evaluation budget.
"""
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import time
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def read_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            break  # A concurrent append may leave an incomplete last line.
    return rows


def stats(values):
    return {'n': len(values), 'median': statistics.median(values),
            'mean': statistics.mean(values), 'sum': sum(values), 'max': max(values)} if values else None


def profile(code, queue):
    from llm4ad.task.optimization.cvrp_aco.dataset import load_split_instances
    from llm4ad.task.optimization.cvrp_aco.evaluation import _distance_matrix
    xs, meta = load_split_instances('train')
    namespace = {}
    exec(code, namespace)
    fn = namespace['heuristics']
    inputs = [(_distance_matrix(x[:, 1:]), x[:, 1:], x[:, 0], meta['capacity']) for x in xs]
    cpu, wall = [], []
    for repeat in range(4):  # One warmup, then three timings on identical inputs.
        cpu0, wall0 = time.process_time(), time.perf_counter()
        for d, coord, dem, cap in inputs:
            prior = fn(d.copy(), coord.copy(), dem.copy(), cap)
            assert np.asarray(prior).shape == d.shape and np.isfinite(prior).all()
        if repeat:
            cpu.append(time.process_time() - cpu0)
            wall.append(time.perf_counter() - wall0)
    queue.put({'instances': len(inputs), 'cpu_seconds_10_instances': cpu,
               'wall_seconds_10_instances': wall, 'dataset': meta})


def main():
    snapshot = {'captured_at': datetime.now().astimezone().isoformat(), 'runs': {}, 'profiles': {}}
    states = {}
    for repeat in range(1, 4):
        run = ROOT / f'experiments/traceaad_v10_5/results/cvrp_aco/20260905_cvrp_v105_rep{repeat}'
        raw = (run / 'tree_state.json').read_bytes()
        state = json.loads(raw)
        states[repeat] = state
        # Align completed records to the atomic checkpoint.
        events = [r for r in read_rows(run / 'events.jsonl') if r['candidate_id'] <= state['completed_attempts']]
        calls = [r for r in read_rows(run / 'llm_calls.jsonl') if r['candidate_id'] <= state['completed_attempts']]
        evaluations = [r for r in read_rows(run / 'evaluations.jsonl') if r['evaluation_id'] <= state['budget_used']]
        def times(rows, key):
            return {'all': stats([r[key] for r in rows]), 'last20': stats([r[key] for r in rows[-20:]])}
        best = max(state['nodes'], key=lambda n: n['fitness'])
        row = {'path': str(run), 'state_sha256': hashlib.sha256(raw).hexdigest(),
               'budget_used': state['budget_used'], 'best': best, 'nodes': len(state['nodes']),
               'llm_seconds': times(calls, 'seconds'), 'eval_seconds': times(evaluations, 'eval_seconds'),
               'completion_tokens': times([r['usage'] for r in calls if r.get('usage')], 'completion_tokens'),
               'timeouts': [r for r in evaluations if r['reason'] == 'timeout'],
               'events': events, 'evaluations': evaluations}
        snapshot['runs'][repeat] = row
        if repeat == 3:
            for call in calls:
                if call['candidate_id'] in [120, 121, 217, 241, 244]:
                    code = call['response'].split('```python', 1)[1].split('```', 1)[0].strip()
                    (OUT / f'rep3_candidate_{call["candidate_id"]}.py').write_text(code + '\n')
    examples = [('rep2_node363', 2, 363), ('rep3_node158', 3, 158),
                ('rep3_node194_complex', 3, 194), ('rep3_node197_simplified', 3, 197)]
    for name, repeat, node_id in examples:
        node = next(n for n in states[repeat]['nodes'] if n['id'] == node_id)
        (OUT / f'{name}.py').write_text(node['code'] + '\n')
        ctx = mp.get_context('spawn')
        queue = ctx.Queue()
        process = ctx.Process(target=profile, args=(node['code'], queue))
        process.start()
        process.join(20)
        if process.is_alive():
            process.kill()
            process.join()
            result = {'status': 'auxiliary_20s_timeout_including_imports'}
        elif process.exitcode != 0:
            result = {'status': 'auxiliary_error', 'exitcode': process.exitcode}
        else:
            result = {'status': 'ok', **queue.get(timeout=1)}
        queue.close()
        snapshot['profiles'][name] = {'node_id': node_id, 'rep': repeat,
                                      'fitness': node['fitness'], 'code_chars': len(node['code']), **result}
    (OUT / 'snapshot.json').write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    print(snapshot['captured_at'])
    for name, result in snapshot['profiles'].items():
        print(name, result)


if __name__ == '__main__':
    main()

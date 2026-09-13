"""Read-only checkpoint/journal audit; prints a compact JSON snapshot to stdout."""
import hashlib
import json
import math
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
MANIFESTS = {
    'v106': 'experiments/traceaad_v10_6/results/batch_20260906_215231_revised.json',
    'v107': 'experiments/traceaad_v10_7/results/batch_20260907_bounded_formal.json',
}


def records(path):
    with path.open() as stream:
        for line in stream:
            if not line.endswith('\n'):
                break  # An active writer can leave an unfinished final line.
            yield json.loads(line)


def audit():
    now = time.time()
    sessions = set(subprocess.check_output(
        ['tmux', 'list-sessions', '-F', '#S'], text=True).splitlines())
    result = {'captured_at': datetime.now().astimezone().isoformat(), 'runs': []}
    for version, manifest_path in MANIFESTS.items():
        manifest = json.loads((ROOT / manifest_path).read_text())
        for plan in manifest['plan']:
            directory = ROOT / Path(manifest_path).parent / plan['task'] / plan['run_name']
            state = json.loads((directory / 'tree_state.json').read_text())
            cutoff, budget = state['completed_attempts'], state['budget_used']
            events = [r for r in records(directory / 'events.jsonl') if r['candidate_id'] <= cutoff]
            receipts = [r for r in records(directory / 'evaluations.jsonl') if r['evaluation_id'] <= budget]
            nodes = state['nodes']
            issues = []
            if [r['candidate_id'] for r in events] != list(range(1, cutoff + 1)):
                issues.append('event IDs not a contiguous unique prefix')
            if [r['evaluation_id'] for r in receipts] != list(range(1, budget + 1)):
                issues.append('evaluation IDs not a contiguous unique prefix')
            evaluated = {r['evaluation_id']: r for r in events if r.get('evaluation_id') is not None}
            if len(evaluated) != budget:
                issues.append('event evaluation count differs from budget')
            for receipt in receipts:
                event = evaluated.get(receipt['evaluation_id'], {})
                if any(event.get(k) != receipt.get(k) for k in ('candidate_id', 'fitness', 'reason', 'eval_seconds')):
                    issues.append(f"receipt mismatch: {receipt['evaluation_id']}")
            good = {r['node_id']: r for r in events if r['status'] == 'ok'}
            if len(good) != len(nodes):
                issues.append('valid-event/node count differs')
            for node in nodes:
                event = good.get(node['id'], {})
                if any(node.get(k) != event.get(k) for k in ('fitness', 'evaluation_id', 'parent_id', 'donor_id', 'operator')):
                    issues.append(f"node mismatch: {node['id']}")
                if not math.isfinite(node['fitness']):
                    issues.append(f"nonfinite archived fitness: {node['id']}")
            roots = [n for n in nodes if n['parent_id'] is None]
            if len(roots) != 8:
                issues.append(f'root count: {len(roots)}')
            drift = [path for path, digest in state['mechanism']['source_hashes'].items()
                     if not Path(path).exists() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest]
            pending = None
            pending_path = directory / 'pending_candidate.json'
            try:
                p = json.loads(pending_path.read_text())
                pending = {k: p.get(k) for k in ('candidate_id', 'phase', 'llm_attempts', 'summary_attempts')}
                pending['age_seconds'] = round(max(0, now - pending_path.stat().st_mtime), 1)
            except FileNotFoundError:
                pass
            summary_path = directory / 'logs/run_summary.json'
            summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
            config = json.loads((directory / 'run_config.json').read_text())
            by_eval = sorted(nodes, key=lambda n: n['evaluation_id'])
            last_gain, best, gains = None, -math.inf, 0
            for node in by_eval:
                if node['fitness'] > best:
                    best = node['fitness']
                    last_gain = node['evaluation_id']
                    gains += node['parent_id'] is not None
            result['runs'].append({
                'version': version, 'task': plan['task'], 'repeat': plan['repeat'],
                'backend': plan['backend'], 'run_dir': str(directory.relative_to(ROOT)),
                'manifest_status': plan['status'], 'summary_status': summary.get('status'),
                'session_alive': plan['session'] in sessions, 'launch_attempts': plan['attempts'],
                'started_at': state['started_at'], 'finished_at': summary.get('finished_at'),
                'budget_used': budget, 'completed_attempts': cutoff, 'valid_nodes': len(nodes),
                'status_counts': dict(Counter(e['status'] for e in events)),
                'failure_reasons': dict(Counter(e['reason'] for e in events if e.get('reason'))),
                'best': best, 'best_root': max(n['fitness'] for n in roots),
                'last_frontier_gain_eval': last_gain, 'frontier_gains': gains,
                'exact_unique_codes': len({n['code'] for n in nodes}),
                'distinct_fitness': len({n['fitness'] for n in nodes}),
                'top_fitness_count': sum(n['fitness'] == best for n in nodes),
                'pending': pending, 'latest_event_ts': events[-1]['ts'],
                'source_drift': drift, 'integrity_issues': issues,
                'task_eval': config['task_eval'], 'llm': state['mechanism']['llm'],
            })
    return result


if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False, indent=2))

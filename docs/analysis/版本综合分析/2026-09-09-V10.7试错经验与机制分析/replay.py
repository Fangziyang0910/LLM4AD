"""Read completed batches only; no model, evaluator, or run-directory writes.

Run from any directory. --output saves a new snapshot; --verify compares a
snapshot with the local source archives. Raw archives are not bundled.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[4]
MANIFESTS = {
    'v106': 'experiments/traceaad_v10_6/results/batch_20260906_215231_revised.json',
    'v107': 'experiments/traceaad_v10_7/results/batch_20260907_bounded_formal.json',
}
SELECTED = {
    ('v106', 'tsp_construct', 1): [26, 29, 63, 64],
    ('v107', 'tsp_construct', 1): [169, 196, 201, 203, 205, 215, 220, 567],
    ('v107', 'tsp_construct', 2): [904],
    ('v107', 'tsp_construct', 3): [136, 148, 825],
    ('v107', 'online_bin_packing', 1): [119, 130, 177, 195, 197, 741],
    ('v107', 'online_bin_packing', 2): [61],
    ('v107', 'op_aco', 3): [830, 844],
    ('v107', 'cvrp_aco', 3): [131, 652],
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replay():
    out = {'captured_at': datetime.now(timezone.utc).isoformat(),
           'source_commit': 'ac6f4b9c', 'runs': [], 'representative_nodes': []}
    for version, manifest in MANIFESTS.items():
        manifest_path = ROOT / manifest
        for plan in json.loads(manifest_path.read_text())['plan']:
            directory = manifest_path.parent / plan['task'] / plan['run_name']
            raw = {name: (directory / name).read_bytes() for name in (
                'tree_state.json', 'events.jsonl', 'evaluations.jsonl',
                'run_config.json', 'logs/run_summary.json')}
            state = json.loads(raw['tree_state.json'])
            events = [json.loads(line) for line in raw['events.jsonl'].splitlines()]
            receipts = [json.loads(line) for line in raw['evaluations.jsonl'].splitlines()]
            config = json.loads(raw['run_config.json'])
            summary = json.loads(raw['logs/run_summary.json'])
            nodes = {n['id']: n for n in state['nodes']}
            evaluated = {e['evaluation_id']: e for e in events if e.get('evaluation_id') is not None}
            assert summary['status'] == 'finished' and state['budget_used'] == 1000
            assert [e['candidate_id'] for e in events] == list(range(1, state['completed_attempts'] + 1))
            assert sorted(evaluated) == [e['evaluation_id'] for e in receipts] == list(range(1, 1001))
            for r in receipts:
                assert all(r.get(k) == evaluated[r['evaluation_id']].get(k)
                           for k in ('candidate_id', 'fitness', 'reason', 'eval_seconds'))
            valid = [e for e in events if e['status'] == 'ok']
            assert len(valid) == len(nodes)
            for e in valid:
                assert all(e.get(k) == nodes[e['node_id']].get(k)
                           for k in ('evaluation_id', 'parent_id', 'donor_id', 'operator', 'fitness'))
            roots = [n for n in nodes.values() if n['parent_id'] is None]
            assert len(roots) == 8
            root_best = max(n['fitness'] for n in roots)
            best = max(nodes.values(), key=lambda n: n['fitness'])
            assert summary['best']['fitness'] == best['fitness']
            source_hashes = {str(Path(p).relative_to(ROOT)): h
                             for p, h in state['mechanism']['source_hashes'].items()}
            # Check the frozen code, not the current V10.7R working tree.
            if version == 'v107':
                for path, expected in source_hashes.items():
                    actual = subprocess.check_output(['git', 'show', f'ac6f4b9c:{path}'], cwd=ROOT)
                    assert digest(actual) == expected, path
            search = [e for e in events if e['parent_id'] is not None]
            for e in search:
                ids = e.get('context_node_ids', [])
                assert all(nodes[i]['evaluation_id'] < (e.get('evaluation_id') or e['budget_used'] + 1) for i in ids)
            curve, frontier, running_best = {}, [], float('-inf')
            for eid, e in evaluated.items():
                if e['status'] == 'ok' and e['fitness'] > running_best:
                    running_best = e['fitness']
                    frontier.append({k: e.get(k) for k in (
                        'evaluation_id', 'node_id', 'parent_id', 'donor_id',
                        'operator', 'fitness', 'context_node_ids')})
                if eid in (8, 25, 50, 100, 250, 500, 750, 1000):
                    curve[str(eid)] = running_best
            assert running_best == best['fitness']
            assert all(a['fitness'] < b['fitness'] for a, b in zip(frontier, frontier[1:]))
            by_operator = {}
            for op in ('Refine', 'Pivot', 'Fuse'):
                rows = [e for e in search if e['operator'] == op]
                ok = [e for e in rows if e['status'] == 'ok']
                by_operator[op] = {
                    'attempts': len(rows), 'evaluated': sum(e.get('evaluation_id') is not None for e in rows),
                    'valid': len(ok),
                    'above_parent': sum(e['fitness'] > e['parent_fitness'] for e in ok),
                    'above_frontier': sum(e['fitness'] > e['best_before'] for e in ok),
                }
                if version == 'v107':
                    by_operator[op]['above_context'] = sum(
                        e['fitness'] > max(nodes[i]['fitness'] for i in e['context_node_ids']) for e in ok)
            calls, stages, usage, models = 0, Counter(), Counter(), Counter()
            call_hash = hashlib.sha256()
            for line in (directory / 'llm_calls.jsonl').open('rb'):
                call_hash.update(line)
                c = json.loads(line)
                calls += 1
                stages[c.get('stage', 'generation')] += 1
                models[c.get('sampling', {}).get('model', 'unknown')] += 1
                for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    usage[k] += (c.get('usage') or {}).get(k, 0) or 0
                usage['missing_usage_records'] += not bool(c.get('usage'))
            contexts = [e for e in search if e.get('context_node_ids')]
            parent_counts = Counter(e['parent_id'] for e in search)
            assert parent_counts == Counter({int(k): v for k, v in state['parent_selection_counts'].items()})
            code_counts = Counter()
            for nid, count in parent_counts.items():
                code_counts[digest(nodes[nid]['code'].encode())] += count
            run = {
                'version': version, 'task': plan['task'], 'repeat': plan['repeat'],
                'run_dir': str(directory.relative_to(ROOT)), 'status': summary['status'],
                'finished_at': summary['finished_at'], 'backend_config': config['backend'],
                'backend_manifest': plan['backend'], 'call_models': dict(models),
                'source_hashes': source_hashes, 'task_eval': config['task_eval'],
                'budget_used': 1000, 'attempts': len(events), 'valid_nodes': len(nodes),
                'status_counts': dict(Counter(e['status'] for e in events)),
                'failure_reasons': dict(Counter(e['reason'] for e in events if e.get('reason'))),
                'best_root': root_best, 'best': best['fitness'], 'best_node': best['id'],
                'best_eval': best['evaluation_id'], 'curve': curve, 'frontier': frontier,
                'distinct_code': len({n['code'] for n in nodes.values()}),
                'distinct_fitness': len({n['fitness'] for n in nodes.values()}),
                'best_fitness_nodes': sum(n['fitness'] == best['fitness'] for n in nodes.values()),
                'selected_parent_nodes': len(parent_counts),
                'largest_parent_node_share': max(parent_counts.values()) / len(search),
                'largest_parent_code_share': max(code_counts.values()) / len(search),
                'operators': by_operator, 'calls': calls, 'call_stages': dict(stages), 'usage': dict(usage),
                'search_prompt_mean': statistics.mean(e['prompt_tokens'] for e in search),
                'eval_seconds_sum': sum(e.get('eval_seconds', 0) or 0 for e in events),
                'source_files': {k: digest(v) for k, v in raw.items()},
            }
            run['source_files']['llm_calls.jsonl'] = call_hash.hexdigest()
            if version == 'v107':
                run['contexts'] = {
                    'count': len(contexts),
                    'program_counts': dict(Counter(str(e['context_program_count']) for e in contexts)),
                    'all_same_fitness': sum(len({nodes[i]['fitness'] for i in e['context_node_ids']}) == 1 for e in contexts),
                    'collapsed_boundaries': sum(len(set(e['quality_boundaries'])) == 1 for e in contexts),
                    'fit_rejections': sum(len(e['reference_fit_rejections']) for e in contexts),
                }
            out['runs'].append(run)
            for nid in SELECTED.get((version, plan['task'], plan['repeat']), []):
                node = nodes[nid]
                event = next(e for e in valid if e['node_id'] == nid)
                exposure = {}
                for cutoff in (250, 1000):
                    future = [e for e in search if e['candidate_id'] > event['candidate_id'] and e['budget_used'] <= cutoff]
                    children = [e for e in future if e['parent_id'] == nid]
                    exposure[str(cutoff)] = {
                        'parent_attempts': len(children),
                        'parent_evaluations': sum(e.get('evaluation_id') is not None for e in children),
                        'reference_exposures': sum(nid in e.get('context_node_ids', []) and e['parent_id'] != nid for e in future),
                        'donor_exposures': sum(e.get('donor_id') == nid for e in future),
                        'best_direct_child': max((e['fitness'] for e in children if e['status'] == 'ok'), default=None),
                    }
                out['representative_nodes'].append({
                    'version': version, 'task': plan['task'], 'repeat': plan['repeat'],
                    **node, 'code_sha256': digest(node['code'].encode()), 'exposure': exposure,
                    'context_node_ids': event.get('context_node_ids', []),
                })
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--output', type=Path)
    group.add_argument('--verify', type=Path)
    args = parser.parse_args()
    result = replay()
    if args.verify:
        saved = json.loads(args.verify.read_text())
        result.pop('captured_at')
        saved.pop('captured_at')
        assert result == saved, 'Snapshot differs from local archives'
        print('PASS: 30 completed runs, receipts, frozen V10.7 sources, costs, curves and representative nodes')
    else:
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        if args.output:
            args.output.write_text(text)
        else:
            print(text, end='')

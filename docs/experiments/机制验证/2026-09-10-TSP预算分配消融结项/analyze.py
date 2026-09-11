"""Read-only extraction and verification of the stopped TSP allocation study.

--snapshot reads local run archives and writes compact evidence and summaries.
The default verifies the checked-in evidence; --check-source also checks the
original local files. No model calls, program execution or evaluator calls.
"""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import statistics as stats


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = REPO / 'experiments/traceaad_v10_8/results'
BATCH = '20260909_v108_allocation_tsp'
FIELDS = (
    'arm seed candidate_id evaluation_id budget_before operator status reason '
    'node_id parent_id donor_id fitness best_so_far quality_ess corrected_ess '
    'ess_target eligible_nodes parent_count_before history_edge_count code_hash prompt_hash '
    'calls input_tokens output_tokens llm_seconds eval_seconds model'
).split()


def read_json(path):
    return json.loads(path.read_text())


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                       allow_nan=False) + '\n')


def extract():
    manifest_path = ROOT / f'batch_{BATCH}.json'
    manifest = read_json(manifest_path)
    metadata = {'batch': BATCH, 'snapshot_at': datetime.now().astimezone().isoformat(),
                'manifest_sha256': sha(manifest_path), 'runs': []}
    compact, programs = [], []
    for plan in manifest['plan']:
        run = ROOT / plan['task'] / plan['run_name']
        state = read_json(run / 'tree_state.json')
        config = read_json(run / 'run_config.json')
        events = read_lines(run / 'events.jsonl')
        receipts = read_lines(run / 'evaluations.jsonl')
        calls = read_lines(run / 'llm_calls.jsonl')
        assert len(events) == state['completed_attempts']
        assert len(receipts) == state['budget_used']
        assert [e['candidate_id'] for e in events] == list(range(1, len(events) + 1))
        assert [r['evaluation_id'] for r in receipts] == list(range(1, len(receipts) + 1))
        assert [(e['candidate_id'], e['evaluation_id'], e['fitness']) for e in events
                if e.get('evaluation_id')] == [
                    (r['candidate_id'], r['evaluation_id'], r['fitness']) for r in receipts]
        by_candidate = defaultdict(list)
        for call in calls:
            by_candidate[call['candidate_id']].append(call)
        assert set(by_candidate) == {e['candidate_id'] for e in events}
        assert all(c.get('usage') and c['usage'].get('total_tokens') is not None for c in calls)
        nodes = {n['id']: n for n in state['nodes']}
        previous_budget, best = 0, None
        for event in events:
            if event['status'] == 'ok':
                node = nodes[event['node_id']]
                assert (node['fitness'], node['evaluation_id']) == (event['fitness'], event['evaluation_id'])
                best = max(best, event['fitness']) if best is not None else event['fitness']
            assert event['best_so_far'] == best
            row = {field: event.get(field) for field in FIELDS}
            row.update(arm=plan['allocation_arm'], seed=plan['seed'], budget_before=previous_budget)
            for field in ('quality_ess', 'corrected_ess', 'ess_target', 'eligible_nodes', 'parent_count_before'):
                row[field] = event.get('selection', {}).get(field)
            cc = by_candidate[event['candidate_id']]
            row.update(calls=len(cc), input_tokens=sum(c['usage']['prompt_tokens'] for c in cc),
                       output_tokens=sum(c['usage']['completion_tokens'] for c in cc),
                       llm_seconds=sum(c['seconds'] for c in cc),
                       model='|'.join(sorted({c['model'] for c in cc})))
            compact.append(row)
            if event.get('evaluation_id'):
                previous_budget = event['evaluation_id']
        assert previous_budget == state['budget_used']
        roots = [n for n in nodes.values() if n['operator'] == 'Init']
        assert len(roots) == 8
        mechanism = state['mechanism']
        pending_path = run / 'pending_candidate.json'
        pending = read_json(pending_path) if pending_path.exists() else None
        best_node = max(nodes.values(), key=lambda n: n['fitness'])
        metadata['runs'].append({
            'arm': plan['allocation_arm'], 'seed': plan['seed'], 'run_name': plan['run_name'],
            'version': state['version'], 'manifest_status_at_snapshot': plan['status'],
            'actual_evaluations': state['budget_used'], 'completed_attempts': state['completed_attempts'],
            'last_event_at': events[-1]['ts'], 'task_eval': config['task_eval'],
            'mechanism': {k: mechanism[k] for k in (
                'budget', 'n_roots', 'count_exponent', 'quality_ess_schedule',
                'operator_probabilities', 'pivot_uniform_probability', 'history_tokens',
                'traj_gens', 'max_context_tokens', 'output_tokens', 'task_contract_hash')},
            'source_hashes': {k.split(f'/runtime_{BATCH}/')[-1]: v
                              for k, v in mechanism['source_hashes'].items()},
            'files': {str((run / name).relative_to(REPO)): sha(run / name)
                      for name in ('run_config.json', 'tree_state.json', 'events.jsonl',
                                   'evaluations.jsonl', 'llm_calls.jsonl')},
            'roots': [{k: n[k] for k in ('id', 'evaluation_id', 'fitness')} for n in roots],
            'pending': {k: pending.get(k) for k in ('candidate_id', 'phase', 'llm_attempts')}
                       if pending else None,
        })
        selected = {best_node['id']}
        # Cases discussed in the report: breakthrough and its immediate sources.
        cases = {('B', 0): [210, 216, 426], ('B', 1): [223, 258, 263, 486],
                 ('C', 2): [354, 366, 370, 377, 430, 524],
                 ('A', 1): [677], ('C', 0): [375, 578, 584, 659, 893]}
        selected.update(cases.get((plan['allocation_arm'], plan['seed']), []))
        for node_id in list(selected):
            selected.update(nodes[node_id][key] for key in ('parent_id', 'donor_id')
                            if nodes[node_id].get(key) is not None)
        programs.extend({'arm': plan['allocation_arm'], 'seed': plan['seed'], **nodes[node_id]}
                        for node_id in sorted(selected))
        if (plan['allocation_arm'], plan['seed']) == ('C', 2):
            event = next(e for e in events if e.get('parent_id') == 370 and e['operator'] == 'Refine')
            prompt = by_candidate[event['candidate_id']][0]['prompt']
            assert hashlib.sha256(prompt.encode()).hexdigest() == event['prompt_hash']
            (HERE / 'C_seed2_n370_refine.prompt.txt').write_text(prompt)
    assert len(metadata['runs']) == 12
    assert len({json.dumps(r['task_eval'], sort_keys=True) for r in metadata['runs']}) == 1
    assert len({json.dumps(r['source_hashes'], sort_keys=True) for r in metadata['runs']}) == 1
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(compact)
    (HERE / 'events.csv.gz').write_bytes(gzip.compress(text.getvalue().encode(), mtime=0))
    write_json('snapshot.json', metadata)
    write_json('programs.json', programs)


def load_rows():
    with gzip.open(HERE / 'events.csv.gz', 'rt', newline='') as stream:
        rows = list(csv.DictReader(stream))
    strings = {'arm', 'operator', 'status', 'reason', 'code_hash', 'prompt_hash', 'model'}
    for row in rows:
        for key, value in row.items():
            if key not in strings:
                row[key] = float(value) if value else None
    return rows


def at_budget(rows, budget):
    end = next(i for i, row in enumerate(rows) if row['evaluation_id'] == budget)
    return rows[:end + 1]


def describe(rows):
    valid = [r for r in rows if r['fitness'] is not None]
    best = max(valid, key=lambda r: r['fitness'])
    evaluated = [r for r in rows if r['evaluation_id'] is not None]
    return {
        'evaluations': len(evaluated), 'candidates': len(rows),
        'length': -best['fitness'], 'best_evaluation': best['evaluation_id'],
        'best_node': best['node_id'], 'calls': sum(r['calls'] for r in rows),
        'input_tokens': sum(r['input_tokens'] for r in rows),
        'output_tokens': sum(r['output_tokens'] for r in rows),
        'llm_seconds': sum(r['llm_seconds'] for r in rows),
        'eval_seconds': sum(r['eval_seconds'] or 0 for r in rows),
        'status': dict(Counter(r['status'] for r in rows)),
        'evaluation_failures': dict(Counter(r['reason'] for r in evaluated if r['status'] != 'ok')),
    }


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['arm'], int(row['seed'])].append(row)
    common_e = min(sum(r['evaluation_id'] is not None for r in run) for run in grouped.values())
    common_calls = int(min(sum(r['calls'] for r in run) for run in grouped.values()))
    runs = []
    for (arm, seed), run in sorted(grouped.items()):
        assert all(r['calls'] == 1 for r in run), 'Call prefix needs intra-candidate handling'
        assert all(math.isfinite(r['fitness']) for r in run if r['fitness'] is not None)
        roots = [r for r in run if r['operator'] == 'Init' and r['status'] == 'ok']
        current = describe(run)
        checkpoints = {str(k): describe(at_budget(run, k))
                       for k in sorted({100, 200, 500, common_e, 750, 1000})
                       if k <= current['evaluations']}
        # Schedule state immediately before a request; the same interval for all runs.
        phase = [r for r in at_budget(run, common_e)
                 if r['budget_before'] >= 200 and r['parent_id'] is not None]
        improvements, best = [], -math.inf
        for row in run:
            if row['fitness'] is not None and row['fitness'] > best:
                best = row['fitness']
                improvements.append({k: row[k] for k in (
                    'candidate_id', 'evaluation_id', 'node_id', 'parent_id',
                    'donor_id', 'operator', 'fitness')})
        phase_frontiers = [r for r in improvements if 200 < r['evaluation_id'] <= common_e]
        curve_mean = stats.mean(-at_budget(run, k)[-1]['best_so_far'] for k in range(200, common_e + 1))
        runs.append({'arm': arm, 'seed': seed, 'init_length': -max(r['fitness'] for r in roots),
                     'init_evaluations': roots[-1]['evaluation_id'], 'checkpoints': checkpoints,
                     'equal_calls': describe(run[:common_calls]), 'current': current,
                     'mean_length_e200_to_common': curve_mean,
                     'phase_e200_to_common': {
                         'attempts': len(phase),
                         'median_base_ess': stats.median(r['corrected_ess'] for r in phase),
                         'median_quality_ess': stats.median(r['quality_ess'] for r in phase),
                         'median_target_ess': stats.median(r['ess_target'] for r in phase),
                         'new_parent_fraction': stats.mean(r['parent_count_before'] == 0 for r in phase),
                         'mean_parent_count': stats.mean(r['parent_count_before'] for r in phase),
                         'frontier_improvements': len(phase_frontiers),
                         'status': dict(Counter(r['status'] for r in phase)),
                     }, 'frontiers': improvements})
    arms = {}
    for arm in 'ABCD':
        rr = [r for r in runs if r['arm'] == arm]
        arms[arm] = {
            'mean_length': {str(k): stats.mean(r['checkpoints'][str(k)]['length'] for r in rr)
                            for k in (200, 500, common_e)},
            'sample_sd_length': {str(k): stats.stdev(r['checkpoints'][str(k)]['length'] for r in rr)
                                 for k in (200, 500, common_e)},
            'mean_init_length': stats.mean(r['init_length'] for r in rr),
            'mean_equal_call_length': stats.mean(r['equal_calls']['length'] for r in rr),
            'mean_curve_length_e200_to_common': stats.mean(r['mean_length_e200_to_common'] for r in rr),
        }
    comparisons = {}
    for left, right in [('B', 'A'), ('C', 'B'), ('D', 'C')]:
        comparisons[f'{left}-{right}'] = {}
        for k in (200, 500, common_e):
            values = [next(r for r in runs if (r['arm'], r['seed']) == (left, seed))['checkpoints'][str(k)]['length']
                      - next(r for r in runs if (r['arm'], r['seed']) == (right, seed))['checkpoints'][str(k)]['length']
                      for seed in range(3)]
            comparisons[f'{left}-{right}'][str(k)] = {'seed_differences': values, 'mean': stats.mean(values)}
    diagnostic_rows = [r for r in at_budget(grouped['C', 2], common_e) if r['parent_id'] == 370]
    parent_fit = next(r['fitness'] for r in grouped['C', 2] if r['node_id'] == 370)
    diagnostic = {
        'run': 'C_seed2', 'parent_node': 370, 'through_evaluation': common_e,
        'attempts': len(diagnostic_rows), 'status': dict(Counter(r['status'] for r in diagnostic_rows)),
        'valid_outcomes': dict(Counter(
            'better' if r['fitness'] > parent_fit else 'equal' if r['fitness'] == parent_fit else 'worse'
            for r in diagnostic_rows if r['fitness'] is not None)),
        'refine_prompt_hash_counts': dict(Counter(r['prompt_hash'] for r in diagnostic_rows
                                                 if r['operator'] == 'Refine')),
    }
    assert sha(HERE / 'C_seed2_n370_refine.prompt.txt') in diagnostic['refine_prompt_hash_counts']
    return {'common_evaluations': common_e, 'common_calls': common_calls,
            'total_evaluations': sum(r['current']['evaluations'] for r in runs),
            'total_calls': sum(r['current']['calls'] for r in runs),
            'runs': runs, 'arms': arms, 'comparisons': comparisons, 'diagnostic': diagnostic}


def plot(rows, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'A': '#64748b', 'B': '#0072b2', 'C': '#009e73', 'D': '#cc79a7'}
    labels = {'A': 'A: current', 'B': 'B: no count penalty',
              'C': 'C: ESS 8', 'D': 'D: ESS 32 to 8'}
    curves = {}
    for arm in 'ABCD':
        for seed in range(3):
            curves[arm, seed] = {int(r['evaluation_id']): -r['best_so_far'] for r in rows
                                if r['arm'] == arm and r['seed'] == seed
                                and r['evaluation_id'] and r['best_so_far'] is not None}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True, layout='constrained')
    common = summary['common_evaluations']
    for index, ax in enumerate(axes.flat):
        if index == 0:
            for arm in 'ABCD':
                xx = range(50, common + 1)
                ax.step(xx, [stats.mean(curves[arm, s][x] for s in range(3)) for x in xx],
                        where='post', color=colors[arm], label=labels[arm])
            ax.set_title('Mean of 3 runs: common evaluation budget')
            ax.set_xlim(50, common)
            ax.legend(fontsize=8, frameon=False, loc='upper right')
        else:
            seed = index - 1
            for arm in 'ABCD':
                curve = curves[arm, seed]
                xx = [x for x in curve if 50 <= x <= common]
                ax.step(xx, [curve[x] for x in xx], where='post', color=colors[arm])
                xx = [x for x in curve if x >= common]
                ax.step(xx, [curve[x] for x in xx], where='post', color=colors[arm],
                        linestyle='--', alpha=.8)
            ax.axvspan(common, 1000, color='#f1f5f9', zorder=-1)
            ax.axvline(common, color='#94a3b8', linewidth=.8)
            ax.set_title(f'Seed {seed}: dashed tails have unequal budgets')
            ax.set_xlim(50, 1000)
        ax.axvline(500, color='#94a3b8', linewidth=.7, linestyle=':')
        ax.set_ylim(6.0, 6.9)
        ax.set_xlabel('Actual evaluations (failures included)')
        ax.set_ylabel('Best train tour length (lower is better)')
        ax.grid(alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(HERE / 'budget_curves.png', dpi=180)
    fig.savefig(HERE / 'budget_curves.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--check-source', action='store_true')
    parser.add_argument('--plot', action='store_true', help='Requires matplotlib; writes PNG/PDF')
    args = parser.parse_args()
    if args.snapshot:
        extract()
    rows = load_rows()
    summary = summarize(rows)
    node_rows = {(r['arm'], int(r['seed']), int(r['node_id'])): r
                 for r in rows if r['node_id'] is not None}
    for node in read_json(HERE / 'programs.json'):
        row = node_rows[node['arm'], node['seed'], node['id']]
        assert row['code_hash'] == hashlib.sha256(node['code'].encode()).hexdigest()
        assert row['fitness'] == node['fitness'] and row['evaluation_id'] == node['evaluation_id']
    if args.snapshot:
        write_json('summary.json', summary)
    else:
        assert summary == read_json(HERE / 'summary.json'), 'Summary differs from compact evidence'
    if args.check_source:
        for run in read_json(HERE / 'snapshot.json')['runs']:
            for path, expected in run['files'].items():
                assert sha(REPO / path) == expected, f'Source changed: {path}'
        for reference in read_json(HERE / 'historical_reference.json'):
            for path, expected in reference['hashes'].items():
                assert sha(REPO / path) == expected, f'Historical source changed: {path}'
            nodes = read_json(REPO / reference['run'] / 'tree_state.json')['nodes']
            for budget, length in reference['checkpoints'].items():
                assert -max(n['fitness'] for n in nodes if n['evaluation_id'] <= int(budget)) == length
    if args.plot:
        plot(rows, summary)
    print(json.dumps({k: summary[k] for k in ('common_evaluations', 'common_calls',
                                             'total_evaluations', 'total_calls', 'arms',
                                             'comparisons')}, ensure_ascii=False, indent=2))
    print('Evidence and summary verified.')


if __name__ == '__main__':
    main()

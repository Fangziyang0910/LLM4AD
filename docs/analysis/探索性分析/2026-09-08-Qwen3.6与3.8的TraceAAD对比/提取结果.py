"""只读现存日志；两套系统前250次真实评价的描述性比较，不是模型消融。"""
import hashlib
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TASKS = ['tsp_construct', 'cvrp_aco', 'op_aco', 'online_bin_packing']
LIMIT = 250


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def summarize(rows):
    evaluated = [r for r in rows if r['called']]
    valid = [r for r in evaluated if r['status'] == 'ok' and finite(r['fitness'])]
    refine = [r for r in evaluated if r['operator'].lower() == 'refine' and finite(r['parent_fitness'])]
    refine_valid = [r for r in refine if r['status'] == 'ok' and finite(r['fitness'])]
    return {
        'recorded_attempts': len(rows), 'evaluations': len(evaluated),
        'valid_evaluations': len(valid), 'valid_rate': len(valid) / len(evaluated),
        'statuses': dict(Counter(r['status'] for r in rows)),
        'refine_evaluations': len(refine), 'refine_valid': len(refine_valid),
        'refine_improved': sum(r['fitness'] > r['parent_fitness'] + 1e-9 for r in refine_valid),
        'refine_equal_score': sum(abs(r['fitness'] - r['parent_fitness']) <= 1e-9 for r in refine_valid),
        'idea_words_median': statistics.median(r['idea_words'] for r in valid),
        'idea_over_100_words': sum(r['idea_words'] > 100 for r in valid),
        'code_chars_median': statistics.median(r['code_chars'] for r in valid),
    }


def main():
    runs, all_rows, sources = [], [], []
    word_checks = []
    for version, pattern in [('v97_q36', 'traceaad_v9_7/results/*/v9_7_20260814_150927_*'),
                             ('v107_q38', 'traceaad_v10_7/results/*/20260907_bounded_formal_*')]:
        for run in sorted((ROOT / 'experiments').glob(pattern)):
            if not run.is_dir():
                continue
            config = json.loads((run / 'run_config.json').read_text())
            if config['task'] not in TASKS:
                continue
            old = version == 'v97_q36'
            path = run / ('artifacts/candidates.jsonl' if old else 'events.jsonl')
            nodes = {} if old else {n['id']: n for n in json.loads((run / 'tree_state.json').read_text())['nodes']}
            rows, count, digest, candidates = [], 0, hashlib.sha256(), {}
            with path.open() as stream:
                for line_no, line in enumerate(stream, 1):
                    digest.update(line.encode())
                    r = json.loads(line)
                    called = bool(r.get('evaluator_called')) if old else r.get('evaluation_id') is not None
                    if called:
                        count += 1
                        if not old:
                            assert r['evaluation_id'] == count, (path, line_no)
                    node = r if old else nodes.get(r.get('node_id'), {})
                    if not old and node:
                        candidates[r['candidate_id']] = node
                    rows.append({
                        'version': version, 'task': config['task'], 'repeat': config['repeat'],
                        'line': line_no, 'eval_count': count, 'called': called,
                        'status': r['status'], 'operator': (r.get('intent') if old else r.get('operator')) or 'Init',
                        'fitness': r.get('child_fitness') if old else r.get('fitness'),
                        'parent_fitness': r.get('parent_fitness'),
                        'idea_words': len((node.get('idea') or '').split()),
                        'code_chars': len((node.get('program') if old else node.get('code')) or ''),
                    })
                    if count == LIMIT:
                        break
            assert count == LIMIT, (path, count)
            sources.append({'path': str(path.relative_to(ROOT)), 'prefix_lines': len(rows), 'prefix_sha256': digest.hexdigest()})
            record = {'version': version, 'task': config['task'], 'repeat': config['repeat'],
                      'source': str(path.relative_to(ROOT)), 'config': config, **summarize(rows)}
            runs.append(record)
            all_rows.extend(rows)
            if not old:
                matched = set()
                for line in (run / 'llm_calls.jsonl').open():
                    call = json.loads(line)
                    if call['candidate_id'] > max(candidates):
                        break
                    n = candidates.get(call['candidate_id'])
                    if n is None or n['id'] in matched or n['idea'] not in (call.get('response') or ''):
                        continue
                    matched.add(n['id'])
                    word_checks.append({'run': run.name, 'node': n['id'], 'evaluation': n['evaluation_id'],
                                        'words': len(n['idea'].split()),
                                        'explicit_100_word_prompt': 'Keep the Idea within 100 words.' in call['prompt']})
                assert len(matched) == len(candidates), run
    assert len(runs) == 24
    groups = []
    for version in ['v97_q36', 'v107_q38']:
        for task in TASKS + ['ALL']:
            rows = [r for r in all_rows if r['version'] == version and (task == 'ALL' or r['task'] == task)]
            groups.append({'version': version, 'task': task, **summarize(rows)})
    heldout = []
    for task in TASKS:
        path = ROOT / 'experiments/traceaad_v9_7/results' / task / 'eval_best_20260815_parentpath/results.json'
        j = json.loads(path.read_text())
        key = next(k for k in ['eval_results_by_size', 'results_by_split', 'eval_results_by_scale'] if k in j)
        for size, result in j[key].items():
            scores = [r['eval_objective'] for r in result['results']]
            assert len(scores) == 3 and all(finite(x) for x in scores)
            mean, sd = statistics.mean(scores), statistics.stdev(scores)
            assert abs(mean - result['summary']['mean_eval_objective']) < 1e-8
            assert abs(sd - result['summary']['sample_std_eval_objective']) < 1e-8
            heldout.append({'task': task, 'size': size, 'values': scores, 'mean': mean, 'sample_sd': sd,
                            'source': str(path.relative_to(ROOT))})
    result = {'extracted_at': datetime.now().astimezone().isoformat(), 'budget_per_run': LIMIT,
              'warning': 'Historical systems differ in prompts, selection, operators, output cap, backends and sampling; not a causal model comparison.',
              'sources': sources, 'runs': runs, 'groups': groups, 'q36_v97_heldout': heldout}
    result['v107_word_constraint'] = {
        'checked': len(word_checks), 'explicit_100_word_prompt': sum(r['explicit_100_word_prompt'] for r in word_checks),
        'above100': sum(r['words'] > 100 for r in word_checks),
        'above120': sum(r['words'] > 120 for r in word_checks),
        'maximum': max(r['words'] for r in word_checks),
        'examples_above130': [r for r in word_checks if r['words'] > 130][:8],
        'definition': 'whitespace-separated tokens in archived Idea; exact text confirmed in corresponding raw response',
    }
    (HERE / '过程统计.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for g in groups:
        print(g)


if __name__ == '__main__':
    main()

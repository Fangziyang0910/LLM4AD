"""Nine isolated evaluations; prepare first, review patches, then pass --evaluate."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/home/fang/code/LLM4AD/LLM4AD')
sys.path.insert(0, str(ROOT))
OUTPUT = Path('/tmp/v107_nan_candidates')
SPECS = [(1, 279, 12), (2, 67, 1), (3, 391, 8)]
REPLACEMENTS = {
    1: ('np.mean(sub_matrix_no_self, axis=1)',
        'np.sum(np.where(np.eye(n, dtype=bool), 0.0, sub_matrix_no_self), axis=1) / (n - 1)'),
    2: ('np.sum(sub, axis=1) / (n_remaining - 1)',
        'np.sum(np.where(np.eye(n_remaining, dtype=bool), 0.0, sub), axis=1) / (n_remaining - 1)'),
    3: ('np.mean(cand_exit_matrix, axis=1)',
        'np.sum(np.where(np.eye(n, dtype=bool), 0.0, cand_exit_matrix), axis=1) / (n - 1)'),
}

def prepare():
    OUTPUT.mkdir(exist_ok=True)
    plans = []
    for repeat, node_id, threshold in SPECS:
        run = ROOT / f'experiments/traceaad_v10_7/results/tsp_construct/20260907_bounded_formal_tsp_v107_rep{repeat}'
        state = json.loads((run / 'tree_state.json').read_text())
        node = next(n for n in state['nodes'] if n['id'] == node_id)
        config = json.loads((run / 'run_config.json').read_text())['task_eval']
        original = node['code']
        old, new = REPLACEMENTS[repeat]
        assert original.count(old) == 1
        repaired = original.replace(old, new)
        function = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == 'select_next_node')
        if repeat == 2:
            cutoff = next(n.lineno - 1 for n in function.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'n_total' for t in n.targets))
        else:
            tail = next(n for n in function.body if isinstance(n, ast.If) and ast.unparse(n.test) == f'n <= {threshold}')
            cutoff = tail.end_lineno
        nearest = '\n'.join(original.splitlines()[:cutoff]) + '\n    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])\n'
        for variant, code in [('original', original), ('repair_mean', repaired), ('nearest_plus_tail', nearest)]:
            compile(code, '<diagnostic-candidate>', 'exec')
            name = f'r{repeat}_n{node_id}_{variant}'
            path = OUTPUT / f'{name}.py'
            path.write_text(code)
            patch = ''.join(difflib.unified_diff(original.splitlines(True), code.splitlines(True), fromfile='original', tofile=variant))
            (OUTPUT / f'{name}.diff').write_text(patch)
            plans.append(dict(repeat=repeat,node_id=node_id,threshold=threshold,variant=variant,path=str(path),archived_fitness=node['fitness'],evaluation_id=node['evaluation_id'],config=config,sha256=hashlib.sha256(code.encode()).hexdigest()))
    (OUTPUT / 'plan.json').write_text(json.dumps(plans, indent=2))
    return plans

def evaluate(plans):
    import numpy as np
    from llm4ad.base.evaluate import SecureEvaluator
    from llm4ad.task.optimization.tsp_construct.evaluation import TSPEvaluation
    class ObservedEvaluation(TSPEvaluation):
        def evaluate_program(self, program_str, callable_func):
            observations = {'calls': 0, 'main_calls': 0, 'main_non_nearest': 0, 'tail_calls': 0, 'tail_non_nearest': 0}
            def observe(current, dest, unvisited, distances):
                chosen = callable_func(current, dest, unvisited, distances)
                nearest = int(unvisited[np.argmin(distances[current, unvisited])])
                observations['calls'] += 1
                section = 'main' if len(unvisited) > self.threshold else 'tail'
                observations[section + '_calls'] += 1
                observations[section + '_non_nearest'] += int(chosen != nearest)
                return chosen
            with np.errstate(invalid='ignore'):
                fitness = super().evaluate_program(program_str, observe)
            return {'fitness': fitness, 'behavior': observations}
    results = []
    for row in plans:
        params = {k:v for k,v in row['config'].items() if k != 'split'}
        assert params == dict(n_instance=16, problem_size=50, seed=2024, timeout_seconds=20)
        evaluator = ObservedEvaluation(**params)
        evaluator.threshold = row['threshold']
        started = time.monotonic()
        outcome = SecureEvaluator(evaluator).evaluate_program_with_details(Path(row['path']).read_text())
        result = {**row, 'seconds':time.monotonic()-started, 'result':outcome.result, 'failure_kind':outcome.failure_kind, 'error':outcome.error}
        results.append(result)
        Path('/tmp/v107_nan_counterfactual.json').write_text(json.dumps(results, indent=2))
        print(row['repeat'], row['variant'], result['result'], result['failure_kind'], result['seconds'], flush=True)

if __name__ == '__main__':
    if '--evaluate' in sys.argv:
        evaluate(json.loads((OUTPUT / 'plan.json').read_text()))
    else:
        prepare()
        print('Prepared nine candidates; inspect repair diffs and nearest-tail boundaries before --evaluate.')

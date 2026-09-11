"""Re-evaluate one archived candidate and two local fixes; never touch search state."""
import hashlib
import json
from pathlib import Path
import signal
import sys
import time

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[3]
RUNTIME = REPO / 'experiments/traceaad_v10_10/results/runtime_20260910_v1010_formal_contextfix'
sys.path.insert(0, str(RUNTIME))
from llm4ad.task.optimization.tsp_construct.evaluation import TSPEvaluation


def timeout(*_):
    raise TimeoutError('20 second formal evaluation limit')


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    run = next(r for r in summary['runs'] if r['version'] == '10_10'
               and r['task'] == 'tsp_construct' and r['repeat'] == 1)
    state = json.loads((REPO / run['snapshot'] / 'tree_state.json').read_text())
    node = next(n for n in state['nodes'] if n['id'] == 337)
    code = node['code']
    # Variant A retains the candidate's first move during local optimization.
    assert code.count('for i in range(1, m - 2):') == 1
    fixed_first = code.replace('for i in range(1, m - 2):', 'for i in range(2, m - 2):')
    # Variant B executes the first move of the plan whose cost was compared.
    assert code.count('        candidate_costs[idx] = total_cost') == 1
    follow_plan = code.replace('        candidate_costs[idx] = total_cost',
                               '        candidates[idx] = tour[1]\n        candidate_costs[idx] = total_cost')
    config = json.loads((REPO / run['source'] / 'run_config.json').read_text())['task_eval']
    signal.signal(signal.SIGALRM, timeout)
    results = []
    for name, program in [('original', code), ('keep_first_fixed', fixed_first),
                          ('return_optimized_first', follow_plan)]:
        namespace = {}
        exec(compile(program, '<archived-candidate>', 'exec'), namespace)
        evaluator = TSPEvaluation(**config)
        start = time.monotonic()
        try:
            signal.alarm(config['timeout_seconds'])
            score = evaluator.evaluate(namespace['select_next_node'])
            result = {'fitness': score}
        except TimeoutError as error:
            result = {'error': str(error)}
        finally:
            signal.alarm(0)
        result.update(variant=name, seconds=time.monotonic()-start,
                      code_sha256=hashlib.sha256(program.encode()).hexdigest(), code=program)
        results.append(result)
        print(name, {k:v for k,v in result.items() if k != 'code'}, flush=True)
    evidence = dict(source=run['source'], node_id=337, archived_fitness=node['fitness'],
                    evaluator_runtime=str(RUNTIME.relative_to(REPO)), task_eval=config,
                    note='Three independent diagnostic re-evaluations, excluded from formal search.',
                    results=results)
    (OUT / 'tsp_semantic_check.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()

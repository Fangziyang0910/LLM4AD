"""Read live journals without running candidates; write a bounded audit snapshot."""
import ast
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'experiments/traceaad_v10_9/results'
OUT = Path(__file__).resolve().parent


def rows(path):
    data = path.read_bytes()
    # An active writer may have an incomplete final line.
    data = data[:data.rfind(b'\n') + 1]
    return [json.loads(line) for line in data.splitlines()], {
        'path': str(path.relative_to(ROOT)), 'bytes': len(data),
        'sha256': hashlib.sha256(data).hexdigest(),
    }


def parser_parts(runtime, task):
    """Use frozen regex and parser itself, without importing experiment services."""
    source = runtime / 'llm4ad/method/traceaad_v10_6/traceaad.py'
    tree = ast.parse(source.read_text())
    regex = next(n for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'CODE_RE' for t in n.targets))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TraceAADV106')
    parser = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_response')
    env = dict(ast=ast, re=re,
               _strip_thinking=lambda s: re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL),
               normalize_code=lambda s: '\n'.join(s.replace('\r\n', '\n').replace('\r', '\n').splitlines()).strip())
    exec(compile(ast.Module(body=[regex, parser], type_ignores=[]), str(source), 'exec'), env)
    template_tree = ast.parse((runtime / f'llm4ad/task/optimization/{task}/template.py').read_text())
    template = next(ast.literal_eval(n.value) for n in template_tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'template_program' for t in n.targets))
    target = next(n for n in ast.parse(template).body if isinstance(n, ast.FunctionDef))
    from types import SimpleNamespace
    owner = SimpleNamespace(_template_func=SimpleNamespace(name=target.name, args=ast.unparse(target.args)))
    return env, owner


def parse_kind(call, env, owner):
    response, finish = call['response'], call['finish_reason']
    assert env['parse_response'](owner, response, finish) is None
    if finish not in ('stop', 'length', 'unknown'):
        return 'finish_reason_rejected'
    match = env['CODE_RE'].fullmatch(env['_strip_thinking'](response))
    if not match:
        return 'idea_code_format'
    code = match[2]
    if re.search(r'^\s*```', code, re.MULTILINE):
        return 'nested_code_fence'
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 'python_syntax'
    targets = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == owner._template_func.name]
    if len(targets) != 1:
        return 'target_function_missing_or_multiple'
    def interface(a):
        return ([x.arg for x in a.posonlyargs], [x.arg for x in a.args],
                [x.arg for x in a.kwonlyargs], a.vararg.arg if a.vararg else None,
                a.kwarg.arg if a.kwarg else None)
    expected = ast.parse(f'def target({owner._template_func.args}):\n pass').body[0]
    if interface(targets[0].args) != interface(expected.args):
        return 'function_signature'
    return 'compile_failure'


def main():
    result = {'started_at': datetime.now().astimezone().isoformat(), 'batches': {}}
    evidence = []
    for batch in ('20260909_v109_initaware', '20260909_v109_formal'):
        manifest = json.loads((BASE / f'batch_{batch}.json').read_text())
        runtime = BASE / f'runtime_{batch}'
        output = {'runs': [], 'journal_prefixes': [], 'planned_runs': len(manifest['plan'])}
        total, reasons, parse_types, finish_types = Counter(), Counter(), Counter(), Counter()
        operators = defaultdict(Counter)
        for plan in manifest['plan']:
            path = BASE / plan['task'] / plan['run_name']
            if not (path / 'events.jsonl').exists():
                continue
            events, stamp = rows(path / 'events.jsonl')
            output['journal_prefixes'].append(stamp)
            assert len({e['candidate_id'] for e in events}) == len(events)
            evals, _ = rows(path / 'evaluations.jsonl')
            by_candidate = {e['candidate_id']: e for e in evals}
            assert len(by_candidate) == len(evals)
            bad = {e['candidate_id']: e for e in events if e['status'] == 'invalid_output'}
            call_errors, found, call_count = 0, set(), 0
            env, owner = parser_parts(runtime, plan['task'])
            # Bound to the file length at opening; do not chase live appends.
            with (path / 'llm_calls.jsonl').open('rb') as handle:
                size = (path / 'llm_calls.jsonl').stat().st_size
                while handle.tell() < size:
                    line_number = call_count + 1
                    line = handle.readline()
                    if not line.endswith(b'\n'):
                        break
                    call = json.loads(line)
                    call_count += 1
                    call_errors += bool(call.get('error'))
                    cid = call['candidate_id']
                    if cid in bad and 'response' in call:
                        assert cid not in found
                        found.add(cid)
                        kind = parse_kind(call, env, owner)
                        parse_types[kind] += 1
                        finish_types[call['finish_reason']] += 1
                        bad[cid]['parse_kind'] = kind
                        bad[cid]['call_line'] = line_number
            assert found == set(bad)
            counts = Counter(e['status'] for e in events)
            per_reasons = Counter(e['reason'] for e in events if e['status'] == 'eval_failed')
            for event in events:
                operators[event['operator']][event['status']] += 1
                if event['status'] in ('ok', 'eval_failed'):
                    receipt = by_candidate[event['candidate_id']]
                    assert all(receipt[k] == event[k] for k in ('evaluation_id', 'fitness', 'reason'))
                else:
                    assert event['candidate_id'] not in by_candidate
                if event['status'] in ('eval_failed', 'invalid_output'):
                    evidence.append(dict(batch=batch, run=plan['run_name'], task=plan['task'],
                        **{k: event.get(k) for k in ('candidate_id', 'operator', 'status', 'reason',
                            'evaluation_id', 'eval_seconds', 'parse_kind', 'call_line')}))
            total.update(counts)
            reasons.update(per_reasons)
            output['runs'].append(dict(task=plan['task'], repeat=plan['repeat'], run=plan['run_name'],
                status=plan['status'], completed_candidates=len(events), counts=dict(counts),
                evaluation_reasons=dict(per_reasons), call_errors=call_errors,
                receipt_count=len(evals), committed_evaluations=counts['ok'] + counts['eval_failed'],
                last_event_ts=events[-1]['ts'] if events else None))
        output.update(counts=dict(total), evaluation_reasons=dict(reasons), parse_types=dict(parse_types),
                      invalid_finish_reasons=dict(finish_types), operators={k: dict(v) for k, v in operators.items()})
        result['batches'][batch] = output
    result['finished_at'] = datetime.now().astimezone().isoformat()
    for name, data in [('summary.json', result), ('failures.json', evidence)]:
        (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

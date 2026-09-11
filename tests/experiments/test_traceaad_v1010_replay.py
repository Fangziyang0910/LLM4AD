from collections import Counter

from experiments.traceaad_v10_10.analysis import replay_parser


def write_run(root, task, run_name, calls, events):
    run = root / task / run_name
    run.mkdir(parents=True)
    (run / 'run_config.json').write_text('{"method": "v1010", "task": "%s"}' % task)
    (run / 'llm_calls.jsonl').write_text(
        ''.join(__import__('json').dumps(c) + '\n' for c in calls))
    (run / 'events.jsonl').write_text(
        ''.join(__import__('json').dumps(e) + '\n' for e in events))


def call(candidate_id, response, stage='generation', repair_of=None, finish='stop'):
    record = {'call_id': f'{candidate_id}:1', 'candidate_id': candidate_id,
              'stage': stage, 'response': response, 'finish_reason': finish}
    if repair_of is not None:
        record['repair_of'] = repair_of
    return record


def ok_response(value=7):
    return f'Idea: constant picker.\n```python\ndef select_next_node(\n        current_node, destination_node, unvisited_nodes, distance_matrix):\n    return {value}\n```'


def node_code(value=7):
    return (f'def select_next_node(\n        current_node, destination_node, unvisited_nodes,'
            f' distance_matrix):\n    return {value}')


def test_replay_classifies_recovery_and_masks(tmp_path):
    import hashlib
    import json
    good_hash = hashlib.sha256(node_code().encode()).hexdigest()
    calls = [
        # 1: plain valid output, evaluated ok.
        call(1, ok_response()),
        # 2: no label but valid code — the old parser rejected it as Missing Idea.
        call(2, 'Here is the program.\n```python\n' + node_code() + '\n```'),
        # 3: repair of 2, valid again.
        call(3, ok_response(8), stage='repair', repair_of=2),
        # 4: valid format but the evaluator failed at runtime.
        call(4, ok_response(9)),
        # 5: two code blocks — rejected then and now.
        call(5, ok_response() + '\n```python\nx = 1\n```'),
        # 6: no label and a syntax error the old parser never reported.
        call(6, 'Intro only.\n```python\ndef select_next_node(\n        current_node, destination_node, unvisited_nodes, distance_matrix):\n    return (\n```'),
    ]
    events = [
        {'candidate_id': 1, 'status': 'ok', 'code_hash': good_hash},
        {'candidate_id': 2, 'status': 'invalid_output',
         'error': 'Missing Idea: paragraph; preserve the code and supply its description.'},
        {'candidate_id': 3, 'status': 'ok', 'repair_of': 2},
        {'candidate_id': 4, 'status': 'eval_failed', 'reason': 'runtime_error',
         'code_hash': hashlib.sha256(node_code(9).encode()).hexdigest()},
        {'candidate_id': 5, 'status': 'invalid_output',
         'error': 'Expected one unambiguous Python code block.'},
        {'candidate_id': 6, 'status': 'invalid_output',
         'error': 'Missing Idea: paragraph; preserve the code and supply its description.'},
    ]
    write_run(tmp_path, 'tsp_construct', 'batch_rep1', calls, events)
    (tmp_path / 'batch_batch.json').write_text(json.dumps(
        {'plan': [{'task': 'tsp_construct', 'run_name': 'batch_rep1'}]}))

    report = replay_parser.replay_batch(tmp_path, 'batch')
    (totals, run) = report['totals'], report['runs'][0]
    assert totals['both_accepted'] == 3 and totals['recovered_by_new_parser'] == 1
    assert totals['missing_idea_masked_real_code_error'] == 1
    assert totals['still_rejected'] == 1
    assert not totals.get('new_parser_rejects_old_valid')
    assert run['new_idea_sources']['tagged'] == 3 and run['new_idea_sources']['prose'] == 1
    assert run['repair_matrix'] == {'invalid_output -> ok': 1}
    assert not run['code_mismatches'] and not run['events_without_response']
    assert run['snapshots'][0]['complete_lines'] == len(events)

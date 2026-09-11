import pytest

from test_traceaad_v105 import FakeLLM, TinyEvaluation, response
from llm4ad.method.traceaad_v10_5.traceaad import read_journal
from llm4ad.method.traceaad_v10_10 import TraceAADV1010


def method(path, llm=None, **kwargs):
    return TraceAADV1010(evaluation=TinyEvaluation(), llm=llm or FakeLLM(), run_dir=path,
                         **{'budget': 1000, 'n_roots': 1, **kwargs})


@pytest.mark.parametrize('text', [
    response(7).replace('Idea:', 'idea:').replace('```python', '```PY'),
    'Here is the candidate.\n' + response(7) + '\nExplanation after the code.',
    response(7).rsplit('```', 1)[0],
])
def test_format_recovery_evaluates_original_code_without_extra_call(tmp_path, text):
    m = method(tmp_path, FakeLLM(text), budget=1)
    m.run()
    assert m.tree.best().fitness == 7 and len(m.llm.calls) == 1
    assert read_journal(m.events_path)[0]['parse_mode'] == 'recovered'


@pytest.mark.parametrize('text', [
    response(7) + '\n```python\ndef score(x):\n return 8\n```',
    response(7).replace('score(x)', 'score(y)'),
    response(7).replace('return 7', 'return ('),
    response(7).replace('Idea:', 'Itdea:'),
])
def test_ambiguous_or_invalid_output_is_not_guessed(tmp_path, text):
    assert method(tmp_path).parse_response(text) is None


def test_partial_output_at_token_limit_is_not_salvaged(tmp_path):
    m = method(tmp_path)
    assert m.parse_response(response(7).rsplit('```', 1)[0], 'length') is None
    assert m.parse_response(response(7), 'length') is not None


def test_runtime_failure_repairs_once_and_charges_every_evaluation(tmp_path):
    llm = FakeLLM(response(1), response('missing_name'), response(3))
    m = method(tmp_path, llm, budget=3)
    m.run()
    events, receipts = read_journal(m.events_path), read_journal(m.evaluations_path)
    assert [e['status'] for e in events] == ['ok', 'eval_failed', 'ok']
    assert [e['evaluation_id'] for e in receipts] == [1, 2, 3]
    assert events[1]['error_type'] == 'NameError' and 'missing_name' in events[1]['error']
    assert events[1]['traceback'] and receipts[1]['traceback']
    assert events[2]['repair_of'] == 2 and events[2]['parent_id'] == events[1]['parent_id']
    assert m.parent_selection_counts == {0: 1}
    assert 'missing_name' in llm.calls[2][0] and 'smallest change' in llm.calls[2][0]
    calls = read_journal(m.llm_calls_path)
    assert [c['stage'] for c in calls] == ['generation', 'generation', 'repair']
    assert sum(c['usage']['completion_tokens'] for c in calls) == 60


def test_failed_repair_returns_to_normal_search_and_parse_uses_no_eval(tmp_path):
    m = method(tmp_path, FakeLLM(response(1), 'bad output', 'still bad', response(4)), budget=2)
    m.run()
    events = read_journal(m.events_path)
    assert [e['status'] for e in events] == ['ok', 'invalid_output', 'invalid_output', 'ok']
    assert events[2]['repair_of'] == 2 and 'repair_of' not in events[3]
    assert len(read_journal(m.evaluations_path)) == 2
    assert sum(m.parent_selection_counts.values()) == 2


def test_initial_failure_repair_and_final_budget_boundary(tmp_path):
    m = method(tmp_path / 'initial', FakeLLM(response('1/0'), response(2)), budget=2)
    m.run()
    assert m.tree.best().parent_id is None and len(m.tree.roots) == 1
    assert read_journal(m.events_path)[1]['repair_of'] == 1
    final = method(tmp_path / 'final', FakeLLM(response(1), response('1/0'), response(3)), budget=2)
    final.run()
    assert len(final.llm.calls) == 2 and final.budget_used == 2


def test_repaired_duplicate_is_filtered_without_evaluation(tmp_path):
    m = method(tmp_path, FakeLLM(response(1), response('1/0'), response(1), response(4)), budget=3)
    m.run()
    events = read_journal(m.events_path)
    assert events[2]['repair_of'] == 2 and events[2]['status'] == 'duplicate_code'
    assert events[2]['evaluation_id'] is None and len(read_journal(m.evaluations_path)) == 3


@pytest.mark.parametrize('point', ['failure_event', 'failure_checkpoint', 'repair_selected',
                                   'repair_receipt', 'repair_event', 'repair_checkpoint'])
def test_repair_survives_crashes_without_repeating_evaluations(tmp_path, monkeypatch, point):
    m = method(tmp_path, FakeLLM(response(1), response('missing_name'), response(3)), budget=3)
    append, save, persist = m._append_record, m._save_state, m._persist_pending
    def crash_append(path, record):
        append(path, record)
        if ((point == 'failure_event' and path == m.events_path and record['candidate_id'] == 2) or
                (point == 'repair_event' and path == m.events_path and record['candidate_id'] == 3) or
                (point == 'repair_receipt' and path == m.evaluations_path and record['candidate_id'] == 3)):
            raise KeyboardInterrupt
    def crash_save():
        save()
        if ((point == 'failure_checkpoint' and m.completed_attempts == 2) or
                (point == 'repair_checkpoint' and m.completed_attempts == 3)):
            raise KeyboardInterrupt
    def crash_persist():
        persist()
        if point == 'repair_selected' and m.pending['candidate_id'] == 3 and m.pending['phase'] == 'selected':
            raise KeyboardInterrupt
    monkeypatch.setattr(m, '_append_record', crash_append)
    monkeypatch.setattr(m, '_save_state', crash_save)
    monkeypatch.setattr(m, '_persist_pending', crash_persist)
    with pytest.raises(KeyboardInterrupt):
        m.run()
    needs_call = point in ('failure_event', 'failure_checkpoint', 'repair_selected')
    restored = method(tmp_path, FakeLLM(response(3)) if needs_call else FakeLLM(), budget=3)
    original = restored.secure.evaluate_program_with_details
    evaluated = []
    def evaluate(code):
        evaluated.append(code)
        assert needs_call and 'missing_name' not in code
        return original(code)
    monkeypatch.setattr(restored.secure, 'evaluate_program_with_details', evaluate)
    restored.run()
    assert len(evaluated) == int(needs_call)
    assert restored.budget_used == 3 and restored.tree.best().fitness == 3
    assert restored.parent_selection_counts == {0: 1}
    assert [e['evaluation_id'] for e in read_journal(restored.evaluations_path)] == [1, 2, 3]
    assert len(read_journal(restored.events_path)) == 3
    assert len(read_journal(restored.llm_calls_path)) == 3


def test_repair_keeps_core_error_without_traceback_or_parent_code(tmp_path):
    from llm4ad.method.traceaad_v10_10.errors import repair_prompt
    event = {'operator': 'Refine', 'parent_fitness': 1, 'reason': 'runtime_error',
             'error_type': 'FileNotFoundError',
             'error': "Missing /private/evaluator/input.txt via '/private/evaluator/cache.bin'",
             'traceback': 'Traceback: /private/evaluator/runner.py'}
    text = repair_prompt('TASK', '<think>internal reasoning</think>' + response(3), event)
    assert 'FileNotFoundError' in text and 'Missing' in text
    assert '/private/' not in text and 'Traceback' not in text and 'internal reasoning' not in text
    assert 'return 3' in text and 'Evaluated parent' not in text
    m = method(tmp_path)
    from test_traceaad_v108 import add
    node = add(m.tree, 1)
    m.builder.max_tokens = 1
    assert m.eligible_nodes() == [node]


def test_task_supplied_design_notes_are_added_without_task_name_branch(tmp_path):
    from llm4ad.task.optimization.op_aco import OPACOEvaluation
    m = TraceAADV1010(evaluation=OPACOEvaluation(n_ants=1, n_iterations=1),
                      llm=FakeLLM(), run_dir=tmp_path, budget=1, n_roots=1)
    assert '# Evaluator Semantics' in m.task_contract
    assert 'Node 0 is masked as a candidate' in m.task_contract
    assert m.mechanism['task_contract_hash']


@pytest.mark.parametrize('kind', ['prepare_error', 'evaluation_error'])
def test_infrastructure_failure_stops_without_llm_repair(tmp_path, monkeypatch, kind):
    from llm4ad.base.evaluate import EvaluationOutcome
    m = method(tmp_path, FakeLLM(response(1), response(2), response(3)), budget=3)
    m._advance()
    if kind == 'prepare_error':
        monkeypatch.setattr(m.secure, 'evaluate_program_with_details', lambda _:
            EvaluationOutcome(result=None, failure_kind=kind,
                              error_type='OSError', error='worker setup failed'))
    else:
        def fail(_):
            raise OSError('evaluation transport failed')
        monkeypatch.setattr(m.secure, 'evaluate_program_with_details', fail)
    with pytest.raises(RuntimeError, match='evaluation infrastructure failed'):
        m.run()
    events = read_journal(m.events_path)
    assert events[-1]['reason'] == kind
    assert len(m.llm.calls) == 2


@pytest.mark.parametrize('name', [
    'test_transport_retry_restores_same_parent_rng_and_selection_count',
    'test_persisted_response_survives_crash_before_evaluator',
    'test_unknown_evaluation_blocks_instead_of_redrawing_or_refunding',
])
def test_inherited_execution_contract(tmp_path, monkeypatch, name):
    import test_traceaad_v105 as contract
    monkeypatch.setattr(contract, 'method', method)
    getattr(contract, name)(*((tmp_path,) if 'transport' in name else (tmp_path, monkeypatch)))


@pytest.mark.parametrize('name', [
    'test_initialization_observes_prior_roots_without_bootstrap',
    'test_initial_copy_rejection_does_not_reject_equal_score_new_code',
])
def test_copied_v109_search_semantics(tmp_path, monkeypatch, name):
    import test_traceaad_v109 as contract
    monkeypatch.setattr(contract, 'method', method)
    getattr(contract, name)(tmp_path)


@pytest.mark.parametrize('kind', ['timeout', 'invalid_result', 'nonfinite_fitness'])
def test_evaluation_failure_types_keep_diagnostics_and_repair_cap(tmp_path, monkeypatch, kind):
    from llm4ad.base.evaluate import EvaluationOutcome
    m = method(tmp_path, FakeLLM(response(1), response(2), response(3), response(4)), budget=4)
    m._advance()
    if kind == 'nonfinite_fitness':
        outcome = EvaluationOutcome(result=float('nan'))
    else:
        outcome = EvaluationOutcome(result=None, failure_kind=kind,
            error_type='TimeoutError' if kind == 'timeout' else 'InvalidEvaluationResult',
            error='evaluation exceeded 20s' if kind == 'timeout' else 'evaluator returned None')
    monkeypatch.setattr(m.secure, 'evaluate_program_with_details', lambda _: outcome)
    m.run()
    events = read_journal(m.events_path)
    assert [e.get('repair_of') for e in events] == [None, None, 2, None]
    assert [e['reason'] for e in events[1:]] == [kind] * 3
    assert len(read_journal(m.evaluations_path)) == 4
    assert 'total time limit' in m.llm.calls[2][0] if kind == 'timeout' else events[1]['error']


def test_transport_failure_during_repair_resumes_the_same_request(tmp_path):
    m = method(tmp_path, FakeLLM(response(1), response('missing'), RuntimeError('offline')), budget=3)
    with pytest.raises(RuntimeError, match='offline'):
        m.run()
    restored = method(tmp_path, FakeLLM(response(3)), budget=3)
    restored.run()
    assert restored.llm.calls[0][0] == m.llm.calls[2][0]
    assert restored.parent_selection_counts == {0: 1}
    assert [c['call_id'] for c in read_journal(m.llm_calls_path)] == ['1:1', '2:1', '3:1', '3:2']
    assert read_journal(m.events_path)[-1]['repair_of'] == 2


@pytest.mark.parametrize('operator', ['Refine', 'Tune', 'Fuse', 'Pivot'])
def test_new_structure_has_no_birth_or_attempt_bonus(tmp_path, operator):
    from test_traceaad_v108 import add
    m = method(tmp_path)
    nodes = [add(m.tree, i) for i in range(20)]
    new = add(m.tree, -20, nodes[0].id, code='def score(x):\n    return x * 0.01', operator='Pivot')
    nodes.append(new)
    m.budget_used = new.evaluation_id
    before, stats = m.node_distribution(nodes, operator)
    assert stats['quality_ess'] == pytest.approx(8)
    assert before[-1] < before[0]
    m.parent_selection_counts[new.id] = 100
    m.budget_used += 100
    assert m.node_distribution(nodes, operator)[0] == before
    assert not any(k.startswith('development_') for k in m.mechanism)
    if operator == 'Pivot':
        assert min(before) >= .5 / len(nodes)


@pytest.mark.parametrize('operator', ['Refine', 'Tune', 'Fuse', 'Pivot'])
def test_child_trials_do_not_change_generation_context(tmp_path, operator):
    from test_traceaad_v108 import add
    m = method(tmp_path)
    root = add(m.tree, 1)
    host = add(m.tree, 2, root.id)
    donor = add(m.tree, 4, root.id) if operator == 'Fuse' else None
    before = m.builder.build(host, operator, donor)
    add(m.tree, 0, host.id, code='def score(x):\n    return x - 500', operator='Tune')
    add(m.tree, 300, host.id, code='def score(x):\n    return x + 300')
    after = m.builder.build(host, operator, donor)
    assert after == before
    text, meta = after
    edges = {(r['source_id'], r['target_id'], r['role']) for r in meta['evidence_relations']}
    if operator in ('Tune', 'Pivot'):
        assert not edges and meta['history_ids'] == []
        assert 'history_tokens' not in meta
        assert meta['context_node_ids'] == [host.id]
        assert 'Formation History' not in text
        if operator == 'Pivot':
            assert '# Reference Algorithm' in text
    else:
        assert (root.id, host.id, 'Host formation') in edges
    if donor:
        assert (root.id, donor.id, 'Donor formation') in edges
    assert 'direct trials' not in text.lower()
    assert meta['context_best_fitness'] == (4 if donor else 2)


def test_operator_recipe_is_used_by_scheduler_and_journal(tmp_path):
    m = method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    m.run()
    expected = {'Refine': .25, 'Tune': .25, 'Fuse': .25, 'Pivot': .25}
    assert m.mechanism['operator_probabilities'] == expected
    assert read_journal(m.events_path)[-1]['operator_probabilities'] == expected


@pytest.mark.parametrize('repair', [False, True])
def test_long_input_uses_remaining_output_space(tmp_path, repair):
    m = method(tmp_path, FakeLLM(response(1)), budget=1)
    m.pending = m._schedule()
    m.pending['prompt'] = 'input ' * 18000
    m.pending['prompt_tokens'] = m.builder.count(m.pending['prompt'], chat=True)
    if repair:
        m.pending['repair_of'] = 0
    m.builder.check_capacity(m.pending['prompt'])
    m._generate_pending()
    expected = 32768 - 256 - m.pending['prompt_tokens']
    assert m.llm.calls[-1][1]['max_tokens'] == expected
    assert read_journal(m.llm_calls_path)[-1]['max_tokens'] == expected
    assert m.output_tokens == 16384


def test_context_is_assembled_once_without_independent_history_quota(tmp_path):
    from test_traceaad_v108 import add
    m = method(tmp_path, n_roots=8)
    roots = [add(m.tree, i) for i in range(3)]
    text, meta = m.builder.build(None, 'Init')
    assert meta['initial_reference_ids'] == [n.id for n in roots]
    assert [text.index(n.code) for n in roots] == sorted(text.index(n.code) for n in roots)
    host = add(m.tree, 4, roots[0].id)
    _, meta = m.builder.build(host, 'Fuse', roots[1])
    assert meta['history_edge_count'] == 1 and not meta['context_omissions']
    assert 'history_tokens' not in meta
    m.builder.max_tokens = 1
    assert m.select_donor(host)[0] is not None
    with pytest.raises(ValueError, match='complete prompt exceeds'):
        m.builder.build(host, 'Fuse', roots[1])


def test_parse_once_and_keep_multiline_exception_message(tmp_path, monkeypatch):
    from llm4ad.method.traceaad_v10_10 import errors
    original = errors.parse_candidate
    calls = []
    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(errors, 'parse_candidate', counted)
    m = method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    m.run()
    assert len(calls) == 2
    assert all('edit_kind' not in e for e in read_journal(m.events_path))
    assert 'ess_fraction' not in m.mechanism and 'ess_minimum' not in m.mechanism
    message = 'expected shape (n,n), got (n,)\nInput values: [1, 2, 3]'
    text = errors.repair_prompt('TASK', response(2), {'operator': 'Tune',
        'reason': 'runtime_error', 'error_type': 'ValueError', 'error': message})
    assert message in text
    long_message = 'core failure\n' + 'x' * (errors.ERROR_MESSAGE_MAX_CHARS + 100)
    text = errors.repair_prompt('TASK', response(2), {'operator': 'Tune',
        'reason': 'runtime_error', 'error_type': 'ValueError', 'error': long_message})
    assert 'core failure' in text
    assert 'x' * errors.ERROR_MESSAGE_MAX_CHARS not in text

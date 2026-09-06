from __future__ import annotations

import json
import pytest

from llm4ad.base import Evaluation
from llm4ad.method.traceaad_v10_3.schema import Node
from llm4ad.method.traceaad_v10_5.prompts import PromptBuilder, formation_events, render_history
from llm4ad.method.traceaad_v10_5.traceaad import TraceAADV105, UnknownEvaluation, read_journal


class TinyEvaluation(Evaluation):
    def __init__(self):
        super().__init__(template_program="def score(x):\n    pass",
                         task_description="Return a numeric score.", safe_evaluate=False)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(1)


class FakeLLM:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def count_tokens(self, text):
        return len(text.split())

    def count_prompt_tokens(self, text):
        return self.count_tokens(text) + 7

    def draw_sample_with_details(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response if isinstance(response, dict) else {
            'content': response, 'finish_reason': 'stop', 'usage': {'completion_tokens': 20},
        }


def response(value=1):
    return f"Idea: Return the requested constant.\nThis tests actual execution.\n```python\ndef score(x):\n    return {value}\n```"


def method(path, llm=None, **kwargs):
    return TraceAADV105(evaluation=TinyEvaluation(), llm=llm or FakeLLM(), run_dir=path,
                        **{'budget': 10, 'n_roots': 1, **kwargs})


def add(m, q, parent=None, code='def score(x):\n    return 1'):
    return m.tree.add(code=code, idea=f'idea {len(m.tree.nodes)}', fitness=q,
                      evaluation_id=len(m.tree.nodes) + 1, parent_id=parent,
                      operator='Init' if parent is None else 'Refine')


def test_history_aligns_child_idea_and_parent_score_and_never_duplicates_current():
    root = Node(0, 'root_code', 'root idea', 1)
    middle = Node(1, 'middle_code', 'middle idea', 3, parent_id=0, operator='Pivot')
    current = Node(2, 'complete_current_code', 'current idea', 2, parent_id=1, operator='Refine')
    edges = formation_events(current, [middle, root], 8)
    assert [n.id for n, _ in edges] == [1, 2]
    history = render_history(edges, set())
    assert 'Pivot | Fitness: 1 -> 3 | improved\nIdea: middle idea' in history
    assert 'Refine | Fitness: 3 -> 2 | regressed\nIdea: current idea' in history
    builder = PromptBuilder(FakeLLM(), 'TASK', max_tokens=1000, history_tokens=2048, max_events=8)
    prompt = builder.build(current, [middle, root], 'Fuse', root)
    assert prompt.text.count('current idea') == 1
    assert prompt.text.index('# Reference') < prompt.text.index('# Formation') < prompt.text.index('# Design Direction')
    assert 'complete_current_code' in prompt.text
    assert prompt.tokens == len(prompt.text.split()) + 7


def test_history_drops_oldest_whole_event_and_omits_oversized_idea():
    root = Node(0, 'root_code', 'root idea', 1)
    middle = Node(1, 'middle_code', 'old ' * 200, 3, parent_id=0, operator='Pivot')
    current = Node(2, 'complete_current_code', 'recent idea', 2, parent_id=1, operator='Refine')
    builder = PromptBuilder(FakeLLM(), 'TASK', max_tokens=1000, history_tokens=50, max_events=8)
    prompt = builder.build(current, [middle, root], 'Refine')
    assert '[Design text omitted due to length.]' in prompt.text
    assert 'idea_too_long:1' in prompt.omissions
    builder.history_tokens = 20
    prompt = builder.build(current, [middle, root], 'Refine')
    assert prompt.history_ids == (2,)
    assert 'history_budget:1' in prompt.omissions
    assert 'complete_current_code' in prompt.text
    minimum = builder.count(builder.assemble(current, 'Refine', None, [], set()), chat=True)
    builder.max_tokens = minimum
    prompt = builder.build(current, [middle, root], 'Refine')
    assert prompt.history_ids == () and prompt.tokens == minimum
    builder.max_tokens -= 1
    with pytest.raises(ValueError, match='minimum complete'):
        builder.build(current, [middle, root], 'Refine')


def test_operator_conditioned_distribution_and_count_penalty(tmp_path):
    m = method(tmp_path)
    nodes = [add(m, 1) for _ in range(3)]
    m.parent_selection_counts[0] = 8
    p0, pivot, stats = m.parent_distribution(nodes, 'Pivot')
    assert p0 == pytest.approx([1/7, 3/7, 3/7])
    assert pivot == pytest.approx([0.5*p + 0.5/3 for p in p0])
    assert stats['quality_ess'] == pytest.approx(3)
    assert stats['corrected_ess'] < stats['conditional_ess'] < 3
    assert m.parent_distribution(nodes, 'Fuse')[1] == p0


def test_requested_operator_frequencies_and_fuse_fallback_keep_pivot_mass(tmp_path):
    m = method(tmp_path)
    add(m, 1)
    counts = {'Refine': 0, 'Pivot': 0, 'Fuse': 0}
    actual = counts.copy()
    for _ in range(4000):
        p = m._schedule()
        counts[p['requested_operator']] += 1
        actual[p['operator']] += 1
        if p['requested_operator'] == 'Fuse':
            assert p['operator'] == 'Refine' and p['donor_id'] is None
            assert p['selection']['fallback_reason']
    assert counts['Refine']/4000 == pytest.approx(.50, abs=.03)
    assert counts['Pivot']/4000 == pytest.approx(.15, abs=.03)
    assert counts['Fuse']/4000 == pytest.approx(.35, abs=.03)
    assert actual['Pivot'] == counts['Pivot'] and actual['Fuse'] == 0
    assert m.parent_selection_counts == {0: 4000}


def test_context_eligibility_and_topk_donors_after_fit_filter(tmp_path):
    m = method(tmp_path, donor_topk=2)
    root = add(m, 1)
    parent = add(m, 2, root.id)
    child = add(m, 100, parent.id)
    huge = add(m, 1000, code='huge ' * 2000)
    other = [add(m, q) for q in [10, 9, 8]]
    m.builder.max_tokens = 300
    assert huge not in m.eligible_nodes()
    assert m.tree.best() == huge
    donors = m.fitting_donors(parent)
    assert donors == other[:2]
    assert root not in donors and child not in donors


@pytest.mark.parametrize('signature', ['other(x)', 'score(y)', 'score(x, *, missing)', 'score(x, *args)', 'score(x, /)'])
def test_invalid_target_interface_is_not_evaluated(tmp_path, signature):
    m = method(tmp_path)
    assert m.parse_response(f'Idea: Test\n```python\ndef {signature}:\n    return 1\n```') is None


def test_helpers_and_multiline_idea_execute_in_one_call(tmp_path):
    text = 'Idea: Delegate to a helper.\nReturn its numeric result.\n```python\ndef helper(x):\n    return x + 6\n\ndef score(x):\n    return helper(x)\n```'
    llm = FakeLLM(text)
    m = method(tmp_path, llm, budget=1)
    m.run()
    assert m.tree.best().fitness == 7
    assert '\nReturn its' in m.tree.best().idea
    assert len(llm.calls) == 1 and llm.calls[0][1]['max_tokens'] == 16384
    assert json.loads(m.summary_path.read_text())['status'] == 'finished'


@pytest.mark.parametrize('code', [
    'def score(x):\n    return x + OFFSET\n\nOFFSET = 7',
    'def shift(fn):\n    return lambda x: fn(x) + 7\n\n@shift\ndef score(x):\n    return x',
    'def helper(x):\n    return Offset.value + x\n\nclass Offset:\n    value = 7\n\ndef score(x):\n    return helper(x)',
])
def test_archived_module_is_exactly_the_evaluated_module(tmp_path, code):
    m = method(tmp_path, FakeLLM(f'Idea: Keep module semantics.\n```python\n{code}\n```'), budget=1)
    m.run()
    assert m.tree.best().fitness == 8
    assert m.tree.best().code == code
    assert m.parse_response(f'Idea: Keep semantics.\n```python\n{code}\n```')[2] == code
    counts = read_journal(tmp_path / 'tokenizer_calls.jsonl')
    assert counts and all('tokens' in r and 'seconds' in r and 'text_hash' in r for r in counts)


def test_budget_counts_actual_failures_but_not_length_or_parse_failure(tmp_path):
    llm = FakeLLM({'content': response(), 'finish_reason': 'length'}, 'bad output',
                  response(4), response('float("nan")'), response('1/0'), response(4), response(2))
    m = method(tmp_path, llm, budget=5)
    m.run()
    assert len(llm.calls) == 7 and m.budget_used == 5
    assert [n.fitness for n in m.tree.all_nodes()] == [4, 4, 2]
    events = read_journal(m.events_path)
    assert [r['budget_used'] for r in events] == [0, 0, 1, 2, 3, 4, 5]
    assert events[0]['reason'] == 'length_truncated'
    assert events[3]['reason'] == 'nonfinite_fitness'
    assert len(read_journal(m.evaluations_path)) == 5
    assert events[-1]['parent_delta'] == events[-1]['fitness'] - events[-1]['parent_fitness']


def test_transport_retry_restores_same_parent_rng_and_selection_count(tmp_path):
    llm = FakeLLM(response(), RuntimeError('connection lost'))
    m = method(tmp_path, llm, budget=2)
    with pytest.raises(RuntimeError, match='connection lost'):
        m.run()
    pending = json.loads(m.pending_path.read_text())
    resumed_llm = FakeLLM(response(2))
    resumed = method(tmp_path, resumed_llm, budget=2)
    resumed.run()
    assert resumed_llm.calls[0][0] == llm.calls[1][0]
    assert resumed.parent_selection_counts == {pending['parent_id']: 1}
    assert resumed.budget_used == 2 and len(resumed_llm.calls) == 1
    assert [r['call_id'] for r in read_journal(resumed.llm_calls_path)] == ['1:1', '2:1', '2:2']


def test_persisted_response_survives_crash_before_evaluator(tmp_path, monkeypatch):
    m = method(tmp_path, FakeLLM(response(3)), budget=1)
    def crash(parsed):
        raise KeyboardInterrupt
    monkeypatch.setattr(m, '_evaluate_pending', crash)
    with pytest.raises(KeyboardInterrupt):
        m.run()
    assert json.loads(m.pending_path.read_text())['phase'] == 'responded'
    resumed = method(tmp_path, budget=1)
    resumed.run()
    assert resumed.tree.best().fitness == 3 and resumed.llm.calls == []


@pytest.mark.parametrize('crash_point', ['after_receipt', 'after_event', 'after_checkpoint'])
def test_receipts_and_candidate_commits_are_idempotent(tmp_path, monkeypatch, crash_point):
    m = method(tmp_path, FakeLLM(response(5)), budget=1)
    original_append = m._append_record
    original_save = m._save_state
    def append(path, record):
        original_append(path, record)
        if ((crash_point == 'after_receipt' and path == m.evaluations_path) or
                (crash_point == 'after_event' and path == m.events_path)):
            raise KeyboardInterrupt
    def save():
        original_save()
        if crash_point == 'after_checkpoint' and m.budget_used:
            raise KeyboardInterrupt
    monkeypatch.setattr(m, '_append_record', append)
    monkeypatch.setattr(m, '_save_state', save)
    with pytest.raises(KeyboardInterrupt):
        m.run()
    resumed = method(tmp_path, budget=1)
    def no_second_evaluation(parsed):
        pytest.fail('a completed evaluation was repeated')
    monkeypatch.setattr(resumed.secure, 'evaluate_program_with_details', no_second_evaluation)
    resumed.run()
    assert resumed.budget_used == 1 and len(resumed.tree.nodes) == 1
    assert len(read_journal(m.events_path)) == len(read_journal(m.evaluations_path)) == 1
    assert not resumed.pending_path.exists() and resumed.llm.calls == []


def test_unknown_evaluation_blocks_instead_of_redrawing_or_refunding(tmp_path, monkeypatch):
    m = method(tmp_path, FakeLLM(response()), budget=1)
    def die_during_eval(parsed):
        raise KeyboardInterrupt
    monkeypatch.setattr(m.secure, 'evaluate_program_with_details', die_during_eval)
    with pytest.raises(KeyboardInterrupt):
        m.run()
    resumed = method(tmp_path, budget=1)
    with pytest.raises(UnknownEvaluation):
        resumed.run()
    summary = json.loads(resumed.summary_path.read_text())
    assert summary['status'] == 'blocked' and resumed.budget_used == 0
    assert resumed.llm.calls == [] and resumed.pending['evaluation_id'] == 1
    assert summary['confirmed_evaluations'] == 0 and summary['unknown_reservation_count'] == 1
    assert summary['unknown_reservations'] == [{'candidate_id': 1, 'evaluation_id': 1}]


def test_checkpoint_refuses_configuration_drift(tmp_path):
    method(tmp_path, FakeLLM(response()), budget=1).run()
    with pytest.raises(ValueError, match='configuration'):
        method(tmp_path, budget=2).run()


def test_journal_repairs_only_partial_tail_and_preserves_complete_record(tmp_path):
    path = tmp_path / 'journal.jsonl'
    path.write_bytes(b'{"id": 1}\n{"id":')
    assert read_journal(path) == [{'id': 1}]
    assert path.read_bytes() == b'{"id": 1}\n'
    path.write_bytes(b'{"id": 1}')
    assert read_journal(path) == [{'id': 1}]
    assert path.read_bytes().endswith(b'\n')
    path.write_bytes(b'{bad}\n{"id": 1}\n')
    with pytest.raises(ValueError, match='corrupt journal'):
        read_journal(path)

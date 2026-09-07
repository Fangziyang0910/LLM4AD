from __future__ import annotations
import json
import re
import pytest
from llm4ad.base import Evaluation
from llm4ad.method.traceaad_v10_3.schema import Node
from llm4ad.method.traceaad_v10_5.traceaad import UnknownEvaluation, read_journal
from llm4ad.method.traceaad_v10_6.traceaad import TraceAADV106, joint_parent_distribution
from llm4ad.method.traceaad_v10_6.prompts import (
    PromptBuilder, formation_events, INSTRUCTIONS, OUTPUT, SUMMARY_INSTRUCTION,
    build_task_contract, build_summary_prompt,
)


@pytest.mark.parametrize('p0', [[1.], [0., 1.], [.2, .3, .5], [1/8]*8])
def test_parent_first_joint_probabilities(p0):
    marginal, conditional = joint_parent_distribution(p0)
    assert sum(marginal) == pytest.approx(1)
    for p, m, c in zip(p0, marginal, conditional):
        assert sum(c.values()) == pytest.approx(1)
        assert m*c['Refine'] == pytest.approx(.5*p)
        assert m*c['Pivot'] == pytest.approx(.15*(.5*p+.5/len(p0)))
        assert m*c['Fuse'] == pytest.approx(.35*p)


@pytest.mark.parametrize('tail,finish,status', [
    ('Idea: Actual implementation.', 'stop', 'present'),
    (None, 'stop', 'unavailable'),
    ('', 'stop', 'unavailable'), ('unlabelled explanation', 'unknown', 'unavailable'),
    ('Idea: unfinished', 'length', 'truncated')])
def test_complete_module_is_evaluated_despite_summary_failure(tmp_path, tail, finish, status):
    code = 'def helper(x):\n    return OFFSET + x\n\ndef score(x):\n    return helper(x)\n\nOFFSET = 7'
    llm = FakeLLM(f'Idea: Planned implementation.\n```python\n{code}\n```',
                  summaries=[{'content': tail, 'finish_reason': finish}])
    m = method(tmp_path, llm, budget=1)
    m.run()
    assert m.tree.best().fitness == 8 and m.tree.best().code == code
    assert len(llm.calls) == 2 and m.budget_used == 1
    event = read_journal(m.events_path)[0]
    assert event['summary_status'] == status
    assert bool(m.tree.best().idea) == (status == 'present')
    assert json.loads(m.state_path.read_text())['version'] == 106


@pytest.mark.parametrize('text,finish', [
    ('Idea: plan\n```python\ndef score(x):\n    return 1', 'length'),
    ('```python\ndef score(x):\n    return 1\n```', 'stop'),
    ('Idea: plan\n```python\ndef score(y):\n    return 1\n```', 'stop'),
    ('Idea: plan\n```python\ndef score(x):\n    return 1\n```\n```python\npass\n```', 'stop'),
    ('Idea: plan\n```python\ndef score(x):\n    return 1\n```', 'content_filter')])
def test_bad_code_or_ambiguous_output_consumes_no_evaluation(tmp_path, text, finish):
    m = method(tmp_path, FakeLLM({'content': text, 'finish_reason': finish}, response(4)), budget=1)
    m.run()
    assert m.budget_used == 1 and len(m.llm.calls) == 3
    assert len(read_journal(m.evaluations_path)) == 1
    assert read_journal(m.events_path)[0]['status'] == 'invalid_output'


def test_failures_and_duplicate_code_consume_formal_slots(tmp_path):
    m = method(tmp_path, FakeLLM(response(4), response('1/0'), response('float("nan")'), response(4)), budget=4)
    m.run()
    assert m.budget_used == 4 and len(m.tree.nodes) == 2
    assert [e['status'] for e in read_journal(m.events_path)] == ['ok','eval_failed','eval_failed','ok']


def test_history_contains_actual_reference_scores_and_no_controller_labels():
    root = Node(0,'root_code','root summary',1)
    ref = Node(3,'ref_code','ref summary',5)
    child = Node(1,'child_code','implemented child',3,parent_id=0,operator='Fuse',donor_id=3)
    b = PromptBuilder(FakeLLM(), 'TASK', max_tokens=1000, history_tokens=8192, max_events=8,
                      lookup={3:ref}.get)
    prompt = b.build(child,[root],'Refine')
    assert prompt.history_ids == (1,)
    assert 'Previous version fitness: 1' in prompt.text
    assert 'Reference algorithm fitness: 5' in prompt.text
    assert prompt.text.count('implemented child') == 1
    assert all(x not in prompt.text for x in ['Operator:', 'donor_id', 'Refine:', 'Pivot:'])
    assert prompt.history_tokens > 0 and prompt.summaries['1']['status']=='present'
    assert 'Idea: <design idea>' in prompt.text
    assert 'approximately 500 words' not in prompt.text
    calibration = build_summary_prompt('TASK', 'New plan', child.code, root)
    assert 'relative to the parent code' in calibration
    assert 'relative to the parent code' not in build_summary_prompt('TASK', 'New plan', root.code)
    assert 'root_code' in calibration and 'child_code' in calibration
    assert 'root summary' not in calibration and 'Fitness:' not in calibration


def test_summary_prefix_preserves_whole_paragraphs_and_minimum_prompt():
    node = Node(0,'root_code','one two\n\n'+'long '*30,1)
    b = PromptBuilder(FakeLLM(),'TASK',max_tokens=1000,history_tokens=8192,max_events=8,summary_tokens=8)
    view, info = b.summary(node)
    assert view == 'one two\n\n[Remaining summary paragraphs omitted.]'
    assert info['view_tokens']<=8 and info['raw_tokens']>8 and info['status']=='prefix'
    huge = Node(1,'huge_code','word '*100,2)
    assert b.summary(huge)[0]=='' and b.summary(huge)[1]['status']=='oversized'
    assert node.idea.endswith('long '*30)
    minimum = b.count(b.assemble(node,'Refine',None,[],{0}),chat=True)
    b.max_tokens=minimum
    assert b.fits(node,'Refine')
    assert 'context_summary:0' in b.build(node,[],'Refine').omissions


def test_history_cap_removes_oldest_whole_events():
    root = Node(0,'root','root summary',1)
    middle = Node(1,'middle','middle summary',2,parent_id=0)
    child = Node(2,'child','child summary',3,parent_id=1)
    b=PromptBuilder(FakeLLM(),'TASK',max_tokens=2000,history_tokens=8192,max_events=8)
    b.history_tokens=b.count(b.render_history([(child,middle)]))
    p=b.build(child,[middle,root],'Pivot')
    assert p.history_ids==(2,) and 'history_budget:1' in p.omissions
    assert 'middle summary' not in p.text and 'child summary' in p.text


def test_positive_task_and_operator_instructions():
    for text in [*INSTRUCTIONS.values(), OUTPUT, SUMMARY_INSTRUCTION]:
        assert not re.search(r'\b(?:not|never|avoid|without)\b',text,re.I)


def test_requested_operator_frequencies_and_fallback(tmp_path):
    m=method(tmp_path); add(m,1)
    counts=dict.fromkeys(['Refine','Pivot','Fuse'],0)
    for _ in range(4000):
        p=m._schedule(); counts[p['requested_operator']]+=1
        assert p['selection']['parent_route']=='joint_marginal'
        if p['requested_operator']=='Fuse': assert p['operator']=='Refine'
    assert [counts[x]/4000 for x in counts]==pytest.approx([.5,.15,.35],abs=.03)
    assert m.parent_selection_counts=={0:4000}

class TinyEvaluation(Evaluation):
    def __init__(self):
        super().__init__(template_program="def score(x):\n    pass",
                         task_description="Return a numeric score.", safe_evaluate=False)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return callable_func(1)

class FakeLLM:
    def __init__(self, *responses, summaries=None):
        self.responses = iter(responses)
        self.summaries = iter(summaries) if summaries is not None else None
        self.calls = []

    def count_tokens(self, text):
        return len(text.split())

    def count_prompt_tokens(self, text):
        return self.count_tokens(text) + 7

    def draw_sample_with_details(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        if '# Implementation Description\n' in prompt:
            response = next(self.summaries) if self.summaries is not None else 'Idea: Calibrated implementation.'
        else:
            response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response if isinstance(response, dict) else {
            'content': response, 'finish_reason': 'stop', 'usage': {'completion_tokens': 20},
        }

def response(value=1, summary='Return the requested value.'):
    return f"Idea: {summary}\n```python\ndef score(x):\n    return {value}\n```"


def method(path, llm=None, **kwargs):
    return TraceAADV106(evaluation=TinyEvaluation(), llm=llm or FakeLLM(), run_dir=path,
                       **{'budget':10, 'n_roots':1, **kwargs})

def add(m, q, parent=None, code='def score(x):\n    return 1'):
    return m.tree.add(code=code, idea=f'idea {len(m.tree.nodes)}', fitness=q,
                      evaluation_id=len(m.tree.nodes) + 1, parent_id=parent,
                      operator='Init' if parent is None else 'Refine')

def test_transport_retry_restores_same_parent_rng_and_selection_count(tmp_path):
    llm = FakeLLM(response(), RuntimeError('connection lost'))
    m = method(tmp_path, llm, budget=2)
    with pytest.raises(RuntimeError, match='connection lost'):
        m.run()
    pending = json.loads(m.pending_path.read_text())
    resumed_llm = FakeLLM(response(2))
    resumed = method(tmp_path, resumed_llm, budget=2)
    resumed.run()
    assert resumed_llm.calls[0][0] == llm.calls[2][0]
    assert resumed.parent_selection_counts == {pending['parent_id']: 1}
    assert resumed.budget_used == 2 and len(resumed_llm.calls) == 2
    assert [r['call_id'] for r in read_journal(resumed.llm_calls_path)] == [
        '1:1', '1:summary:1', '2:1', '2:2', '2:summary:1']

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

@pytest.mark.parametrize("crash_point", ["after_receipt", "after_event", "after_checkpoint"])
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


@pytest.mark.parametrize('task,digest', [
    ('tsp_construct', '83ec73775c6225f70c69b8619a24304a6922622c920627eaa9b9d69617cd4896'),
    ('cvrp_aco', '57c5d37437823c7a556ca7b2423411a20e684093167016de4b399d7152cae867'),
    ('op_aco', '65d6122fd2428d755a02a755d65830a1cbb73cd12f93e76f3dc42a484a2a6a9c'),
    ('online_bin_packing', '4dc33e33b667251cbda2bec405dd689d0c199b9ce5c7524aaa01565207720a32'),
    ('vrptw_construct', '8a863a7ed9ad48f17e4300f748bf76c40a892477b8cc5d2f8a234b6cb91cfdb6'),
])
def test_shared_task_information_matches_historical_comparison(task, digest):
    import ast
    import hashlib
    import importlib
    from types import SimpleNamespace
    module = importlib.import_module(f'llm4ad.task.optimization.{task}.template')
    # Frozen from the shared definitions matched to the 75 historical baseline runs.
    assert hashlib.sha256((module.task_description+'\0'+module.template_program).encode()).hexdigest() == digest
    evaluation = SimpleNamespace(task_description=module.task_description,
                                 template_program=module.template_program, timeout_seconds=123456)
    contract = build_task_contract(evaluation)
    assert module.task_description.strip() in contract
    function = next(n for n in ast.parse(module.template_program).body if isinstance(n, ast.FunctionDef))
    assert ' '.join(ast.get_docstring(function).split()) in ' '.join(contract.split())
    assert '123456' not in contract
    evaluation.task_description = 'A replacement shared task description.'
    assert evaluation.task_description in build_task_contract(evaluation)
    assert module.task_description not in build_task_contract(evaluation)


def test_calibrated_idea_replaces_plan_in_history_and_calls_are_separate(tmp_path):
    llm = FakeLLM(response(1, 'Original plan.'), response(2, 'Next plan.'),
                  summaries=['Idea: Actual root calculation.', 'Idea: Actual child calculation.'])
    m = method(tmp_path, llm, budget=2)
    m.run()
    assert [n.idea for n in m.tree.all_nodes()] == ['Actual root calculation.', 'Actual child calculation.']
    assert 'Actual root calculation.' in llm.calls[2][0]
    assert 'Original plan.' not in llm.calls[2][0]
    assert 'Next plan.' in llm.calls[3][0] and '# Parent Code' in llm.calls[3][0]
    assert m.task_contract in llm.calls[3][0]
    calls = read_journal(m.llm_calls_path)
    assert [r['stage'] for r in calls] == ['generation', 'thought_alignment'] * 2
    assert len(read_journal(m.evaluations_path)) == 2
    assert all(r['usage']['completion_tokens'] == 20 for r in calls)


def test_summary_transport_failure_preserves_evaluable_code(tmp_path):
    m = method(tmp_path, FakeLLM(response(3), summaries=[RuntimeError('summary connection lost')]), budget=1)
    m.run()
    assert m.tree.best().fitness == 3 and m.tree.best().idea == ''
    assert read_journal(m.events_path)[0]['summary_status'] == 'unavailable'
    assert 'summary connection lost' in read_journal(m.llm_calls_path)[1]['error']
    assert len(read_journal(m.evaluations_path)) == 1


@pytest.mark.parametrize('stage', ['before_summary', 'during_summary', 'after_summary_receipt'])
def test_summary_resume_uses_saved_code_and_saved_description(tmp_path, monkeypatch, stage):
    llm = FakeLLM(response(3), summaries=[KeyboardInterrupt()] if stage == 'during_summary' else None)
    m = method(tmp_path, llm, budget=1)
    if stage == 'before_summary':
        def crash(parsed):
            raise KeyboardInterrupt
        monkeypatch.setattr(m, '_calibrate_pending', crash)
    elif stage == 'after_summary_receipt':
        original = m._log_call
        def crash(record):
            if record.get('stage') == 'thought_alignment':
                raise KeyboardInterrupt
            original(record)
        monkeypatch.setattr(m, '_log_call', crash)
    with pytest.raises(KeyboardInterrupt):
        m.run()
    resumed = method(tmp_path, budget=1)
    resumed.run()
    assert resumed.tree.best().fitness == 3
    assert len(resumed.llm.calls) == (0 if stage == 'after_summary_receipt' else 1)
    assert all('# Implementation Description' in call[0] for call in resumed.llm.calls)
    assert len(read_journal(m.evaluations_path)) == 1


def test_oversized_summary_context_keeps_full_code_without_another_request(tmp_path):
    m = method(tmp_path, FakeLLM(response(3)), budget=1)
    m.pending = m._schedule()
    m._generate_pending()
    m.builder.max_tokens = 1
    m._advance()
    assert m.tree.best().fitness == 3 and m.tree.best().idea == ''
    assert len(m.llm.calls) == 1
    assert read_journal(m.events_path)[0]['summary_status'] == 'context_exceeded'


def test_legacy_single_call_checkpoint_is_not_resumed_as_new_protocol(tmp_path):
    m = method(tmp_path, budget=1)
    m._save_state()
    state = json.loads(m.state_path.read_text())
    state['mechanism']['generation'] = 'single_call_code_then_summary'
    m.state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='configuration'):
        method(tmp_path, budget=1).run()


def test_closed_code_at_generation_length_limit_still_gets_calibrated(tmp_path):
    m = method(tmp_path, FakeLLM({'content': response(7), 'finish_reason': 'length'}), budget=1)
    m.run()
    assert m.tree.best().fitness == 7 and m.tree.best().idea == 'Calibrated implementation.'
    assert len(m.llm.calls) == 2 and len(read_journal(m.evaluations_path)) == 1

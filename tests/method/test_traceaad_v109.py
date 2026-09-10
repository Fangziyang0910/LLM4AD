import json
import pytest

from test_traceaad_v105 import FakeLLM, TinyEvaluation, response
from test_traceaad_v108 import add
from llm4ad.method.traceaad_v10_5.traceaad import read_journal
from llm4ad.method.traceaad_v10_9 import TraceAADV109
from llm4ad.method.traceaad_v10_9.traceaad import code_key


def method(path, llm=None, **kwargs):
    return TraceAADV109(evaluation=TinyEvaluation(), llm=llm or FakeLLM(), run_dir=path,
                        **{'budget': 1000, 'n_roots': 1, **kwargs})


def test_concentrated_quality_and_bounded_development(tmp_path):
    m = method(tmp_path)
    nodes = [add(m.tree, i) for i in range(100)]
    m.budget_used = 100
    base, stats = m.node_distribution(nodes, 'Refine')
    assert stats['quality_ess'] == pytest.approx(8)
    m.parent_selection_counts = {n.id: 100 for n in nodes}
    assert m.node_distribution(nodes, 'Refine')[0] == base
    new = add(m.tree, -20, nodes[0].id, code='def score(x):\n    return x * 0.01', operator='Pivot')
    m.budget_used = new.evaluation_id
    p, stats = m.node_distribution(m.tree.all_nodes(), 'Refine')
    assert p[-1] >= 0.2 and stats['development_node_ids'] == [new.id]
    m.parent_selection_counts[new.id] = 2
    assert m.node_distribution(m.tree.all_nodes(), 'Refine')[1]['development_node_ids'] == []
    m.parent_selection_counts[new.id] = 0
    m.budget_used += 64
    assert m.node_distribution(m.tree.all_nodes(), 'Refine')[1]['development_node_ids'] == []
    pivot, stats = m.node_distribution(m.tree.all_nodes(), 'Pivot')
    assert min(pivot) >= .5 / len(pivot) and stats['development_node_ids'] == []


def test_numeric_signature_and_safe_duplicate_scope(tmp_path):
    a = 'def score(x):\n    return x * -1.0'
    b = '# new comment\ndef score(x):\n    return x * 2.0'
    assert code_key(a, True) == code_key(b, True)
    assert code_key(a) != code_key(b)
    assert code_key('x = True', True) != code_key('x = 2', True)
    assert code_key('x = "1"', True) != code_key('x = "2"', True)
    assert code_key('def f():\n    "doc A"\n    return 1') != code_key('def f():\n    "doc B"\n    return 1')
    m = method(tmp_path)
    parent = add(m.tree, 1, code=a)
    donor = add(m.tree, 2, code=b)
    m.pending = {'operator': 'Fuse', 'parent_id': parent.id, 'donor_id': donor.id}
    assert m._duplicate_inputs(b.replace('# new comment', '# renamed'))[0]['role'] == 'donor'
    assert m._duplicate_inputs(b.replace('2.0', '3.0')) == []


def test_task_evidence_contains_direct_trials_and_donor_formation(tmp_path):
    m = method(tmp_path)
    root = add(m.tree, 1)
    host = add(m.tree, 2, root.id)
    bad = add(m.tree, 0, host.id, code='def score(x):\n    return x - 5', operator='Tune')
    good = add(m.tree, 3, host.id)
    donor = add(m.tree, 4, root.id, code='def score(x):\n    return x * 4')
    text, meta = m.builder.build(host, 'Fuse', donor)
    edges = {(r['source_id'], r['target_id'], r['role']) for r in meta['evidence_relations']}
    assert (host.id, bad.id, 'Direct trial') in edges
    assert (host.id, good.id, 'Direct trial') in edges
    assert (root.id, donor.id, 'Donor formation') in edges
    assert 'wholesale donor replacement' in text and 'return x - 5' in text
    assert meta['context_best_fitness'] == 4
    tune, _ = m.builder.build(host, 'Tune')
    assert 'Change only numeric constants' in tune
    minimum = m.builder.count(m.builder.assemble(host, 'Fuse', donor), chat=True)
    m.builder.max_tokens = minimum
    compact, stats = m.builder.build(host, 'Fuse', donor)
    assert m.builder.count(compact, chat=True) == minimum
    assert not stats['evidence_relations'] and stats['context_omissions']
    assert stats['context_node_ids'] == [host.id, donor.id]
    m.builder.max_tokens -= 1
    with pytest.raises(ValueError, match='minimum complete'):
        m.builder.build(host, 'Fuse', donor)


def test_copy_rejection_budget_and_resume_actual_tune_classification(tmp_path):
    llm = FakeLLM(response(1), response(1).replace('return 1', '# cosmetic\n    return 1'), response(2), response(3))
    m = method(tmp_path, llm, budget=3)
    m.OPERATOR_PROBABILITIES = {'Tune': 1.0}
    m._advance()
    m._advance()
    assert m.budget_used == 1 and m.completed_attempts == 2
    m._advance()
    assert m.budget_used == 2
    events = read_journal(m.events_path)
    assert events[1]['status'] == 'duplicate_code' and events[1]['evaluation_id'] is None
    assert events[2]['edit_kind'] == 'numeric_only'
    restored = method(tmp_path, FakeLLM(response(3)), budget=3)
    restored.OPERATOR_PROBABILITIES = {'Tune': 1.0}
    restored._load_state()
    assert restored.budget_used == 2 and restored.completed_attempts == 3
    restored._advance()
    assert restored.budget_used == 3
    assert [r['evaluation_id'] for r in read_journal(m.evaluations_path)] == [1, 2, 3]
    assert json.loads(m.state_path.read_text())['version'] == 1091


@pytest.mark.parametrize('name', [
    'test_transport_retry_restores_same_parent_rng_and_selection_count',
    'test_persisted_response_survives_crash_before_evaluator',
    'test_unknown_evaluation_blocks_instead_of_redrawing_or_refunding',
])
def test_inherited_execution_contract(tmp_path, monkeypatch, name):
    import test_traceaad_v105 as contract
    monkeypatch.setattr(contract, 'method', method)
    getattr(contract, name)(*( (tmp_path,) if 'transport' in name else (tmp_path, monkeypatch)))


@pytest.mark.parametrize('crash_point', ['after_receipt', 'after_event', 'after_checkpoint'])
def test_idempotent_crash_recovery(tmp_path, monkeypatch, crash_point):
    import test_traceaad_v105 as contract
    monkeypatch.setattr(contract, 'method', method)
    contract.test_receipts_and_candidate_commits_are_idempotent(tmp_path, monkeypatch, crash_point)


def test_historical_transfer_source_is_visible_and_counted(tmp_path):
    m = method(tmp_path)
    root = add(m.tree, 1)
    donor = add(m.tree, 99, code='def score(x):\n    return x * 99')
    host = add(m.tree, 2, root.id, donor=donor.id, operator='Fuse')
    text, meta = m.builder.build(host, 'Refine')
    assert 'Historical transfer source' in text and donor.code in text
    assert donor.id in meta['context_node_ids'] and meta['context_best_fitness'] == 99


def test_tune_minimum_prompt_is_included_in_parent_eligibility(tmp_path):
    m = method(tmp_path)
    node = add(m.tree, 1)
    old_limit = max(m.builder.count(m.builder.assemble(node, op, None), chat=True)
                    for op in ('Refine', 'Pivot'))
    assert m.builder.count(m.builder.assemble(node, 'Tune', None), chat=True) > old_limit
    m.builder.max_tokens = old_limit
    with pytest.raises(ValueError, match='no archived parent fits'):
        m.eligible_nodes()


def test_failed_tune_still_reports_actual_edit_kind(tmp_path):
    m = method(tmp_path, FakeLLM(response(1), response(1).replace('return 1', 'return 1 / 0')))
    m.OPERATOR_PROBABILITIES = {'Tune': 1.0}
    m._advance()
    m._advance()
    event = read_journal(m.events_path)[-1]
    assert event['status'] == 'eval_failed' and event['edit_kind'] == 'structural'
    assert m.budget_used == 2


def test_initialization_observes_prior_roots_without_bootstrap(tmp_path):
    m = method(tmp_path, FakeLLM(response(1), response(2)), n_roots=2)
    m._advance()
    pending = m._schedule()
    assert pending['operator'] == 'Init' and pending['parent_id'] is None
    assert pending['initial_reference_ids'] == [0]
    assert 'Previous Initial Algorithms' in pending['prompt']
    assert m.tree.nodes[0].code in pending['prompt']
    assert 'another promising decision hypothesis' in pending['prompt']
    m.pending = pending
    m._persist_pending()
    m._advance()
    assert len(m.tree.roots) == 2 and m.budget_used == 2
    assert all(n.parent_id is None for n in m.tree.all_nodes())
    assert m._schedule()['operator'] != 'Init'


def test_initial_copy_rejection_does_not_reject_equal_score_new_code(tmp_path):
    text = 'Idea: Use an equivalent expression.\n```python\ndef score(x):\n    return x * 1\n```'
    m = method(tmp_path, FakeLLM(response(1), response(1), text), n_roots=2)
    m._advance()
    m._advance()
    assert m.budget_used == 1 and len(m.tree.roots) == 1
    e = read_journal(m.events_path)[-1]
    assert e['status'] == 'duplicate_code' and e['reason'] == 'identical_to_existing_root'
    m._advance()
    assert m.budget_used == 2 and len(m.tree.roots) == 2
    assert len({n.fitness for n in m.tree.all_nodes()}) == 1


def test_initial_context_crops_whole_roots_and_restores_selected_prompt(tmp_path):
    m = method(tmp_path, n_roots=8)
    a = add(m.tree, 1)
    b = add(m.tree, 2)
    full, metadata = m.builder.build(None, 'Init')
    assert metadata['initial_reference_ids'] == [a.id, b.id]
    assert a.code in full and b.code in full
    m.builder.max_tokens = m.builder.count(m.builder.assemble(None, 'Init', None), chat=True)
    compact, meta = m.builder.build(None, 'Init')
    assert meta['initial_reference_ids'] == [] and len(meta['context_omissions']) == 2
    assert a.code not in compact and b.code not in compact
    m.builder.max_tokens = 16000
    m._save_state()
    m.pending = m._schedule()
    m._persist_pending()
    prompt = m.pending['prompt']
    restored = method(tmp_path, n_roots=8)
    restored._load_state()
    assert restored.pending['prompt'] == prompt
    assert restored.pending['initial_reference_ids'] == [a.id, b.id]

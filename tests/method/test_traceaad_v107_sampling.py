import json
import random

import pytest

from llm4ad.method.traceaad_v10_3.schema import Node
from llm4ad.method.traceaad_v10_5.traceaad import read_journal
from llm4ad.method.traceaad_v10_7.prompts import TrajectoryBuilder
from llm4ad.method.traceaad_v10_7.sampling import (
    quality_layers, sample_task_evidence,
)
from test_traceaad_v107 import FakeLLM, method, response


def node(index, fitness, parent_id=None):
    return Node(index, f'def score(x):\n    return {index}', f'Idea {index}', fitness,
                parent_id=parent_id)


def task_sample(nodes, parent, operator, seed=0, **kwargs):
    kwargs.setdefault('fits', lambda refs, donor, roles: True)
    return sample_task_evidence(
        nodes, parent, random.Random(seed), operator=operator, limit=2, **kwargs,
    )


def test_ties_share_one_quality_layer():
    nodes = [node(i, 7) for i in range(6)]
    layers, bounds = quality_layers(nodes)
    assert bounds == [7, 7] and set(layers.values()) == {'low'}


def builder():
    return TrajectoryBuilder(FakeLLM(), 'TASK', max_tokens=1000,
                             history_tokens=1, max_events=0)


def test_sorted_full_programs_preserve_explicit_roles_and_donor():
    b = builder()
    parent, weaker, stronger = node(20, 7), node(21, 6), node(22, 8)
    parent.code += '\n# retained original comment'
    for operator in ['Refine', 'Pivot', 'Fuse']:
        donor = stronger if operator == 'Fuse' else None
        roles = {parent.id: 'design_base', weaker.id: 'evidence_reference',
                 stronger.id: 'transfer_source' if donor else 'evidence_reference'}
        text, nodes, donor, executed, _, _ = b.trajectory(
            parent, [stronger, weaker], operator, donor, roles,
        )
        assert [n.id for n in nodes] == [21, 20, 22]
        assert all(text.count(n.code) == text.count(n.idea) == 1 for n in nodes)
        assert executed == operator
        assert all(label not in text for label in ['Previous version', 'Resulting version',
                                                  'Development History', 'parent_id', 'donor_id'])
        instruction = text.split('# Algorithm Design Task\n')[1].split('# Output')[0]
        assert 'Algorithm 2' in instruction
        if operator == 'Fuse':
            assert donor is stronger and 'Algorithm 3' in instruction


def test_capacity_preserves_full_base_and_reduces_material_count():
    b = builder()
    parent, other = node(0, 1), node(1, 2)
    base = b.trajectory(parent, [], 'Refine')[0]
    b.max_tokens = b.count(base, chat=True)
    refs, _, info = task_sample(
        [parent, other], parent, 'Fuse',
        fits=lambda refs, donor, roles: b.fits_references(
            parent, refs, 'Fuse', donor,
        ),
    )
    assert refs == [] and info['reference_shortfall'] == 2
    text, _, donor, executed, _, _ = b.trajectory(parent, refs, 'Fuse')
    assert executed == 'Refine' and donor is None and parent.code in text
    assert b.count(text, chat=True) <= b.max_tokens
    long_parent = node(2, 1)
    long_parent.idea = ' too long' * 1000
    assert b.fits(long_parent, 'Refine')
    assert long_parent.idea not in b.trajectory(long_parent, [], 'Refine')[0]


def test_short_idea_is_omitted_if_only_code_fits():
    b = builder()
    parent = node(0, 1)
    parent.idea = 'design ' * 200
    full = b.trajectory(parent, [], 'Refine')[0]
    b.max_tokens = b.count(full, chat=True) - 150
    text = b.trajectory(parent, [], 'Refine')[0]
    assert parent.code in text and parent.idea not in text
    assert b.fits(parent, 'Refine')


def test_temporary_algorithm_references_are_removed_only_from_prompt_view():
    b = builder()
    archived_code = (
        'def score(x):\n'
        '    # Reuse Algorithm 3 here\n'
        '    label = "Algorithm 3"\n'
        '    return x\n'
    )
    parent = Node(0, archived_code, 'Combine Algorithm 2 with a local rule.', 1)
    text, _, _, _, _, omissions = b.trajectory(
        parent, [], 'Refine', roles={parent.id: 'design_base'},
    )
    assert parent.code == archived_code
    assert parent.idea == 'Combine Algorithm 2 with a local rule.'
    assert '# Reuse Algorithm 3 here' not in text
    assert 'label = "Algorithm 3"' in text
    assert parent.idea not in text
    assert {(item['kind'], item['reason']) for item in omissions} == {
        ('idea', 'temporary_algorithm_reference'),
        ('code_comment', 'temporary_algorithm_reference'),
    }


def test_end_to_end_single_calls_and_only_parent_counts(tmp_path):
    llm = FakeLLM(*(response(i, f'plan {i}') for i in range(1, 7)))
    m = method(tmp_path, llm, budget=6, n_roots=2)
    m.run()
    events = read_journal(m.events_path)
    assert len(llm.calls) == len(read_journal(m.evaluations_path)) == 6
    assert sum(m.parent_selection_counts.values()) == 4
    assert sum(m.implementation_attempt_counts.values()) == 4
    assert sum(m.generation_condition_counts.values()) == 6
    for event in events:
        assert 'history_ids' not in event and 'history_tokens' not in event
        ids = event['context_node_ids']
        assert len(ids) <= 3 and len(ids) == len(event['context_program_tokens'])
        if event['parent_id'] is not None:
            assert ids[event['context_parent_index'] - 1] == event['parent_id']
        if event['parent_id'] is not None:
            assert event['context_delta'] == event['fitness'] - event['context_best_fitness']
            assert event['parent_code_hash']
        assert len(ids) == len(event['context_program_roles']) == len(event['context_code_hashes'])


def test_task_evidence_assigns_operator_specific_roles():
    predecessor = node(1, 4)
    parent = node(2, 5, parent_id=1)
    child = node(3, 6, parent_id=2)
    unrelated = node(4, 2)
    refs, donor, info = task_sample(
        [predecessor, parent, child, unrelated], parent, 'Refine',
    )
    assert len(refs) == 1 and refs[0].id in {1, 3} and donor is None
    assert info['reference_roles'][str(refs[0].id)] == 'formation_contrast'

    similar = node(5, 7)
    alternative = node(6, 1)
    alternative.code = 'def score(x):\n    if x:\n        return 6\n    return 0'
    refs, donor, info = task_sample([parent, similar, alternative], parent, 'Pivot')
    assert refs == [alternative] and donor is None
    assert info['reference_roles'][str(alternative.id)] == 'alternative_reference'

    low, middle, high = node(7, 1), node(8, 5), node(9, 9)
    low.code = 'def score(x):\n    if x:\n        return 7\n    return 0'
    middle.code = 'def score(x):\n    for _ in range(1):\n        pass\n    return 8'
    high.code = 'def score(x):\n    while False:\n        pass\n    return 9'
    refs, donor, info = task_sample([parent, low, middle, high], parent, 'Fuse')
    assert donor is high and refs[0] is high
    assert info['reference_roles'][str(high.id)] == 'transfer_source'
    assert len(refs) == 2


@pytest.mark.parametrize('operator', ['Refine', 'Pivot', 'Fuse'])
def test_task_evidence_bounds_each_failed_reference_slot(operator):
    parent = node(0, 0)
    nodes = [parent, *(node(index, index) for index in range(1, 100))]
    refs, donor, info = sample_task_evidence(
        nodes, parent, random.Random(0), operator=operator, limit=2,
        fits=lambda refs, donor, roles: False,
    )
    assert refs == [] and donor is None
    assert len(info['reference_attempts']) == 32


@pytest.mark.parametrize('point', ['selected', 'responded', 'receipt'])
def test_resume_reuses_context_and_does_not_repeat_evaluation(tmp_path, monkeypatch, point):
    m = method(tmp_path, FakeLLM(response(1), response(2)), budget=2)
    m._save_state()
    m._advance()
    if point == 'selected':
        def crash():
            raise KeyboardInterrupt
        monkeypatch.setattr(m, '_generate_pending', crash)
    elif point == 'responded':
        def crash(parsed):
            raise KeyboardInterrupt
        monkeypatch.setattr(m, '_evaluate_pending', crash)
    else:
        original = m._append_record
        def crash(path, record):
            original(path, record)
            if path == m.evaluations_path:
                raise KeyboardInterrupt
        monkeypatch.setattr(m, '_append_record', crash)
    with pytest.raises(KeyboardInterrupt):
        m._advance()
    saved = json.loads(m.pending_path.read_text())
    resumed = method(tmp_path, FakeLLM(response(2)), budget=2)
    def no_resample():
        pytest.fail('context was resampled on resume')
    monkeypatch.setattr(resumed, '_schedule', no_resample)
    resumed.run()
    event = read_journal(resumed.events_path)[-1]
    assert event['context_node_ids'] == saved['context_node_ids']
    assert event['prompt_hash'] == saved['prompt_hash']
    assert len(resumed.llm.calls) == (1 if point == 'selected' else 0)
    assert len(read_journal(resumed.evaluations_path)) == 2
    assert sum(resumed.parent_selection_counts.values()) == 1
    assert sum(resumed.implementation_attempt_counts.values()) == 1
    assert sum(resumed.generation_condition_counts.values()) == 2

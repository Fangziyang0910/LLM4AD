import json
import random

import pytest

from llm4ad.method.traceaad_v10_3.schema import Node
from llm4ad.method.traceaad_v10_5.traceaad import read_journal
from llm4ad.method.traceaad_v10_7.prompts import (
    TEMPORARY_REFERENCE_RE, TrajectoryBuilder,
)
from llm4ad.method.traceaad_v10_7.sampling import (
    STRUCTURE_PREFERENCE, _bounded_structure_weights, _task_base_weights,
    quality_layers, sample_task_evidence,
)
from test_traceaad_v107 import FakeLLM, method, response


def node(index, fitness, parent_id=None):
    return Node(index, f'def score(x):\n    return {index}', f'Idea {index}', fitness,
                parent_id=parent_id)


def task_sample(nodes, parent, operator, seed=0, **kwargs):
    kwargs.setdefault('fits', lambda refs, donor, roles, relations: True)
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


def test_role_sections_replace_algorithm_numbers_and_fitness_sorting():
    b = builder()
    parent, weaker, stronger = node(20, 7), node(21, 6), node(22, 8)
    parent.code += '\n# retained original comment'
    stronger.code += '\n# dropped reference comment'
    for operator in ['Refine', 'Pivot', 'Fuse']:
        donor = stronger if operator == 'Fuse' else None
        roles = {parent.id: 'design_base', weaker.id: 'evidence_reference',
                 stronger.id: 'transfer_source' if donor else 'evidence_reference'}
        text, nodes, donor, executed, _, omissions = b.trajectory(
            parent, [stronger, weaker], operator, donor, roles,
        )
        # Fixed role order: base first, then evidence in selection order.
        assert [n.id for n in nodes] == [20, 22, 21]
        assert not TEMPORARY_REFERENCE_RE.search(text)
        assert '# Design Base' in text and 'Design note:' in text
        assert '# retained original comment' in text
        assert '# dropped reference comment' not in text
        assert executed == operator
        assert all(label not in text for label in ['Previous version', 'Resulting version',
                                                  'Development History', 'parent_id', 'donor_id'])
        instruction = text.split('# Design Task\n')[1].split('# Output')[0]
        assert 'Design Base' in instruction or 'Baseline' in instruction
        if operator == 'Fuse':
            assert donor is stronger and 'Transfer Source' in instruction
        assert {(item['kind'], item['reason']) for item in omissions} == {
            ('code_comment', 'reference_comment_strip'),
        }


def test_capacity_preserves_full_base_and_reduces_material_count():
    b = builder()
    parent, other = node(0, 1), node(1, 2)
    base = b.trajectory(parent, [], 'Refine')[0]
    b.max_tokens = b.count(base, chat=True)
    refs, _, info = task_sample(
        [parent, other], parent, 'Fuse',
        fits=lambda refs, donor, roles, relations: b.fits_references(
            parent, refs, 'Fuse', donor, roles, relations,
        ),
    )
    assert refs == [] and info['reference_shortfall'] == 1
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
        '    # Reuse Algorithm #3 and Alg 4 here\n'
        '    label = "Algorithm #3 and Alg 4"\n'
        '    return x\n'
    )
    parent = Node(0, archived_code, 'Combine Alg #2 with a local rule.', 1)
    text, _, _, _, _, omissions = b.trajectory(
        parent, [], 'Refine', roles={parent.id: 'design_base'},
    )
    assert parent.code == archived_code
    assert parent.idea == 'Combine Alg #2 with a local rule.'
    assert '# Reuse Algorithm #3 and Alg 4 here' not in text
    assert 'label = "Algorithm #3 and Alg 4"' in text
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
        assert len(ids) <= 2 and len(ids) == len(event['context_program_tokens'])
        if event['parent_id'] is not None:
            assert ids[event['context_parent_index'] - 1] == event['parent_id']
        if event['parent_id'] is not None:
            assert event['context_delta'] == event['fitness'] - event['context_best_fitness']
            assert event['parent_code_hash']
            assert 'evidence_relations' in event
        assert len(ids) == len(event['context_program_roles']) == len(event['context_code_hashes'])


def test_task_evidence_assigns_operator_specific_roles():
    predecessor = node(1, 4)
    parent = node(2, 5, parent_id=1)
    parent.operator = 'Refine'
    child = node(3, 6, parent_id=2)
    child.operator = 'Pivot'
    unrelated = node(4, 2)
    refs, donor, info = task_sample(
        [predecessor, parent, unrelated], parent, 'Refine',
    )
    assert refs == [predecessor] and donor is None
    assert info['reference_roles'][str(predecessor.id)] == 'formation_evidence'
    assert info['evidence_relations'] == [{
        'kind': 'formation_evidence', 'source_id': 1, 'target_id': 2,
        'operator': 'Refine', 'source_fitness': 4, 'target_fitness': 5,
        'fitness_delta': 1, 'direct_generation_relation': True,
    }]

    refs, donor, info = task_sample([parent, child, unrelated], parent, 'Refine')
    assert refs == [child] and donor is None
    assert info['reference_roles'][str(child.id)] == 'development_evidence'
    assert info['evidence_relations'][0]['source_id'] == parent.id
    assert info['evidence_relations'][0]['target_id'] == child.id
    assert info['evidence_relations'][0]['operator'] == 'Pivot'

    similar = node(5, 7)
    alternative = node(6, 1)
    alternative.code = 'def score(x):\n    if x:\n        return 6\n    return 0'
    refs, donor, info = task_sample([parent, similar, alternative], parent, 'Pivot')
    assert len(refs) == 1 and donor is None
    assert info['reference_roles'][str(refs[0].id)] == 'alternative_reference'
    assert info['evidence_relations'][0]['direct_generation_relation'] is False

    low, middle, high = node(7, 1), node(8, 5), node(9, 9)
    low.code = 'def score(x):\n    if x:\n        return 7\n    return 0'
    middle.code = 'def score(x):\n    for _ in range(1):\n        pass\n    return 8'
    high.code = 'def score(x):\n    while False:\n        pass\n    return 9'
    refs, donor, info = task_sample([parent, low, middle, high], parent, 'Fuse')
    assert donor is refs[0]
    assert info['reference_roles'][str(donor.id)] == 'transfer_source'
    assert len(refs) == 1


def test_refine_prefers_formation_then_same_operator_development():
    predecessor = node(1, 4)
    parent = node(2, 5, parent_id=1)
    parent.operator = 'Refine'
    refine_child = node(3, 6, parent_id=2)
    refine_child.operator = 'Refine'
    pivot_child = node(4, 7, parent_id=2)
    pivot_child.operator = 'Pivot'

    refs, _, info = task_sample(
        [pivot_child, refine_child, predecessor, parent], parent, 'Refine',
    )
    assert refs == [predecessor]
    assert info['reference_roles'] == {str(predecessor.id): 'formation_evidence'}
    assert info['quality_boundaries'] == [] and info['reference_layers'] == {}

    refs, _, info = task_sample(
        [pivot_child, refine_child, parent], parent, 'Refine',
    )
    assert refs == [refine_child]
    assert info['reference_roles'] == {str(refine_child.id): 'development_evidence'}

    refs, _, info = task_sample([pivot_child, parent], parent, 'Refine')
    assert refs == [pivot_child]


def test_refine_filters_relationship_before_deduplicating_code():
    predecessor = node(1, 4)
    parent = node(2, 5, parent_id=1)
    duplicate = node(3, 9)
    duplicate.code = predecessor.code

    refs, _, info = task_sample([duplicate, predecessor, parent], parent, 'Refine')

    assert refs == [predecessor]
    assert info['reference_roles'] == {'1': 'formation_evidence'}


def test_refine_does_not_fill_missing_relational_evidence_from_archive():
    parent, unrelated = node(1, 5), node(2, 9)
    refs, donor, info = task_sample([parent, unrelated], parent, 'Refine')
    assert refs == [] and donor is None
    assert info['reference_shortfall'] == 1


def test_structure_bonus_is_multiplicative_with_full_support():
    parent = node(0, 0)
    same = node(1, 10)
    different = node(2, 1)
    different.code = 'def score(x):\n    if x:\n        return 2\n    return 0'

    for operator in ['Pivot', 'Fuse']:
        layers, _ = quality_layers([same, different])
        base = _task_base_weights([same, different], layers, operator)
        weights = _bounded_structure_weights([same, different], parent, operator, layers)
        assert sum(weights) == pytest.approx(1)
        assert all(weight > 0 for weight in weights)
        # The bonus can only scale the task share, within [(1, 1 + λ)].
        assert base[1] <= weights[1] <= (1 + STRUCTURE_PREFERENCE) * base[1]
    layers, _ = quality_layers([same, different])
    pivot = _bounded_structure_weights([same, different], parent, 'Pivot', layers)
    assert pivot[1] / pivot[0] == pytest.approx(1 + STRUCTURE_PREFERENCE)


def test_lone_structure_different_candidate_cannot_capture_mass():
    parent = node(0, 0)
    peers = [node(index, index) for index in range(1, 1000)]
    worst = node(1000, 0)
    worst.code = 'def score(x):\n    if x:\n        return 1000\n    return 0'
    candidates = [*peers, worst]
    layers, _ = quality_layers(candidates)
    for operator in ['Pivot', 'Fuse']:
        weights = _bounded_structure_weights(candidates, parent, operator, layers)
        assert sum(weights) == pytest.approx(1)
        assert all(weight > 0 for weight in weights)
        assert weights[-1] < 0.01


def test_prompt_states_direction_without_causal_claim():
    b = builder()
    predecessor = node(1, 4)
    parent = node(2, 5, parent_id=1)
    parent.operator = 'Refine'
    refs, _, info = task_sample([predecessor, parent], parent, 'Refine')
    roles = {parent.id: 'design_base', predecessor.id: 'formation_evidence'}
    text = b.trajectory(
        parent, refs, 'Refine', roles=roles,
        relations=info['evidence_relations'],
    )[0]

    assert '# Observed Transition' in text
    assert 'Formation Evidence --Refine--> Design Base' in text
    assert 'Fitness changed from 4 to 5 (delta +1)' in text
    assert 'inspect the code before reusing any changed component' in text
    assert 'do not prove' not in text
    assert not TEMPORARY_REFERENCE_RE.search(text)

    child = node(4, 3, parent_id=parent.id)
    child.operator = 'Fuse'
    child.donor_id = 5
    historical_donor = node(5, 8)
    refs, _, info = task_sample([parent, child, historical_donor], parent, 'Refine')
    roles = {parent.id: 'design_base', child.id: 'development_evidence'}
    reverse_text = b.trajectory(
        parent, refs, 'Refine', roles=roles,
        relations=info['evidence_relations'],
    )[0]
    assert 'Design Base --Fuse--> Development Evidence' in reverse_text
    assert reverse_text != text
    assert 'another program also contributed and is not shown' in reverse_text
    assert 'with fitness 8' not in reverse_text

    alternative = node(3, 6)
    refs, _, info = task_sample([parent, alternative], parent, 'Pivot')
    roles = {parent.id: 'comparison_baseline', refs[0].id: 'alternative_reference'}
    text = b.trajectory(
        parent, refs, 'Pivot', roles=roles,
        relations=info['evidence_relations'],
    )[0]
    assert '# Observed Transition' not in text
    assert 'no direct generation relation' not in text
    assert '# Comparison Baseline' in text and '# Alternative Reference' in text


def test_bare_refine_prompt_mentions_no_contrast():
    b = builder()
    parent = node(1, 5)
    text = b.trajectory(parent, [], 'Refine', roles={parent.id: 'design_base'})[0]
    assert '# Observed Transition' not in text
    assert 'contrast' not in text.lower()
    assert '# Design Base' in text


@pytest.mark.parametrize('operator', ['Refine', 'Pivot', 'Fuse'])
def test_task_evidence_bounds_each_failed_reference_slot(operator):
    parent = node(0, 0)
    nodes = [parent, *(node(index, index, parent_id=0)
                       for index in range(1, 100))]
    refs, donor, info = sample_task_evidence(
        nodes, parent, random.Random(0), operator=operator, limit=2,
        fits=lambda refs, donor, roles, relations: False,
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

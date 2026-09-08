import json
import random

import pytest

from llm4ad.method.traceaad_v10_3.schema import Node
from llm4ad.method.traceaad_v10_5.traceaad import read_journal
from llm4ad.method.traceaad_v10_7.prompts import TrajectoryBuilder
from llm4ad.method.traceaad_v10_7.sampling import quality_layers, sample_references
from test_traceaad_v107 import FakeLLM, method, response


def node(index, fitness, parent_id=None):
    return Node(index, f'def score(x):\n    return {index}', f'Idea {index}', fitness,
                parent_id=parent_id)


def sample(nodes, parent, seed=0, **kwargs):
    return sample_references(nodes, parent, random.Random(seed),
                             **{'limit': 2, 'policy': 'sampled_trajectory_v1',
                                'fits': lambda refs: True, **kwargs})


def test_ties_share_layers_and_distinct_code_keeps_opportunities():
    nodes = [node(i, 7) for i in range(6)]
    layers, bounds = quality_layers(nodes)
    assert bounds == [7, 7] and len(set(layers.values())) == 1
    seen = set()
    for seed in range(40):
        refs, _ = sample(nodes, nodes[0], seed)
        assert len(refs) == 2
        seen.update(n.id for n in refs)
    assert seen == {1, 2, 3, 4, 5}


def test_never_best_ancestor_and_descendant_are_all_sampleable():
    nodes = [node(i, -q, i - 1 if i else None) for i, q in enumerate([10, 9, 8, 6, 7])]
    seen = set()
    for seed in range(60):
        refs, info = sample(nodes, nodes[2], seed)
        seen.update(n.id for n in refs)
        assert len(set(info['reference_layers'].values())) == 2
    assert seen == {0, 1, 3, 4}


def test_duplicate_records_are_selected_intact_without_best_bias():
    parent = node(0, 10)
    first = node(1, 2)
    duplicate = Node(2, first.code, 'other actual record', 3)
    same_as_parent = Node(3, parent.code, 'duplicate parent', 100)
    seen = set()
    for seed in range(30):
        refs, _ = sample([parent, first, duplicate, same_as_parent], parent, seed)
        assert len(refs) == 1
        assert refs[0] is first or refs[0] is duplicate
        seen.add(refs[0].id)
    assert seen == {1, 2}


def test_uniform_control_samples_all_codes_and_is_seeded():
    nodes = [node(i, i) for i in range(8)]
    assert sample(nodes, nodes[0], 4) == sample(nodes, nodes[0], 4)
    seen = set()
    for seed in range(80):
        refs, _ = sample(nodes, nodes[0], seed, policy='uniform_trajectory_v1')
        seen.update(n.id for n in refs)
    assert seen == set(range(1, 8))


def builder():
    return TrajectoryBuilder(FakeLLM(), 'TASK', max_tokens=1000,
                             history_tokens=1, max_events=0)


def test_sorted_full_programs_preserve_parent_role_and_best_sampled_donor():
    b = builder()
    parent, weaker, stronger = node(20, 7), node(21, 6), node(22, 8)
    parent.code += '\n# retained original comment'
    for operator in ['Refine', 'Pivot', 'Fuse']:
        text, nodes, donor, executed, _ = b.trajectory(parent, [stronger, weaker], operator)
        assert [n.id for n in nodes] == [21, 20, 22]
        assert all(text.count(n.code) == text.count(n.idea) == 1 for n in nodes)
        assert executed == operator
        assert all(label not in text for label in ['Previous version', 'Resulting version',
                                                  'Development History', 'parent_id', 'donor_id'])
        instruction = text.split('# Algorithm Design Task\n')[1].split('# Output')[0]
        if operator == 'Pivot':
            assert 'Algorithm 2' not in instruction
        else:
            assert 'Algorithm 2' in instruction
        if operator == 'Fuse':
            assert donor is stronger and 'Algorithm 3' in instruction


def test_capacity_preserves_full_base_and_reduces_material_count():
    b = builder()
    parent, other = node(0, 1), node(1, 2)
    base = b.trajectory(parent, [], 'Refine')[0]
    b.max_tokens = b.count(base, chat=True)
    refs, info = sample([parent, other], parent,
                        fits=lambda refs: b.fits_references(parent, refs, 'Fuse'))
    assert refs == [] and info['reference_fit_rejections'] == [1]
    text, _, donor, executed, _ = b.trajectory(parent, refs, 'Fuse')
    assert executed == 'Refine' and donor is None and parent.code in text
    assert b.count(text, chat=True) <= b.max_tokens
    long_parent = node(2, 1)
    long_parent.idea = ' too long' * 1000
    assert b.fits(long_parent, 'Refine')
    assert long_parent.idea not in b.trajectory(long_parent, [], 'Refine')[0]


def test_bounded_pair_retries_and_unique_attempts():
    nodes = [node(i, 1) for i in range(100)]
    refs, info = sample(nodes, nodes[0], fits=lambda refs: len(refs) <= 1)
    assert len(refs) == 1 and len(info['reference_attempts']) == 33
    ids = [row['node_id'] for row in info['reference_attempts']]
    assert len(ids) == len(set(ids))


def test_zero_references_does_not_tokenize_or_consume_rng():
    rng = random.Random(4)
    before = rng.getstate()
    def forbidden(refs):
        pytest.fail('unneeded capacity request')
    refs, _ = sample_references([node(0, 0), node(1, 1)], node(0, 0), rng,
                                limit=0, policy='sampled_trajectory_v1', fits=forbidden)
    assert refs == [] and rng.getstate() == before


def test_large_archive_never_prefilters_all_candidates():
    nodes = [node(i, i) for i in range(1000)]
    calls = []
    refs, _ = sample(nodes, nodes[0], fits=lambda refs: calls.append(refs) or False)
    assert refs == [] and len(calls) <= 64


def test_short_idea_is_omitted_if_only_code_fits():
    b = builder()
    parent = node(0, 1)
    parent.idea = 'design ' * 200
    full = b.trajectory(parent, [], 'Refine')[0]
    b.max_tokens = b.count(full, chat=True) - 150
    text = b.trajectory(parent, [], 'Refine')[0]
    assert parent.code in text and parent.idea not in text
    assert b.fits(parent, 'Refine')


def test_failed_pair_retries_same_layer_before_switching():
    nodes = [node(i, i) for i in range(10)]
    _, info = sample(nodes, nodes[0], fits=lambda refs: len(refs) == 1)
    failures = info['reference_attempts'][1:]
    # Each layer is exhausted as a contiguous block, rather than reselected.
    runs = []
    for attempt in failures:
        if not runs or runs[-1] != attempt['layer']:
            runs.append(attempt['layer'])
    assert len(runs) == len(set(runs))


@pytest.mark.parametrize('policy', ['sampled_trajectory_v1', 'uniform_trajectory_v1'])
def test_end_to_end_single_calls_and_only_parent_counts(tmp_path, policy):
    llm = FakeLLM(*(response(i, f'plan {i}') for i in range(1, 7)))
    m = method(tmp_path, llm, budget=6, n_roots=2, context_policy=policy)
    m.run()
    events = read_journal(m.events_path)
    assert len(llm.calls) == len(read_journal(m.evaluations_path)) == 6
    assert sum(m.parent_selection_counts.values()) == 4
    for event in events:
        assert 'history_ids' not in event and 'history_tokens' not in event
        ids = event['context_node_ids']
        assert len(ids) <= 3 and len(ids) == len(event['context_program_tokens'])
        if event['parent_id'] is not None:
            assert ids[event['context_parent_index'] - 1] == event['parent_id']
        if event['donor_id'] is not None:
            refs = [m.tree.nodes[i] for i in ids if i != event['parent_id']]
            assert m.tree.nodes[event['donor_id']].fitness == max(n.fitness for n in refs)
        assert event['context_policy'] == policy


@pytest.mark.parametrize('point', ['selected', 'responded', 'receipt'])
def test_resume_reuses_context_and_does_not_repeat_evaluation(tmp_path, monkeypatch, point):
    m = method(tmp_path, FakeLLM(response(1), response(2)), budget=2,
               context_policy='sampled_trajectory_v1')
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
    resumed = method(tmp_path, FakeLLM(response(2)), budget=2,
                     context_policy='sampled_trajectory_v1')
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


def test_context_policy_drift_is_rejected(tmp_path):
    method(tmp_path, FakeLLM(response()), context_policy='ancestor_history').run()
    with pytest.raises(ValueError, match='configuration'):
        method(tmp_path, FakeLLM(), context_policy='sampled_trajectory_v1').run()

import json
import math
import random
import shutil
import subprocess

import pytest

import test_traceaad_v105 as execution_contract
from test_traceaad_v105 import FakeLLM, TinyEvaluation, response
from llm4ad.method.traceaad_v10_3.schema import SearchTree
from llm4ad.method.traceaad_v10_5.traceaad import read_journal
from llm4ad.method.traceaad_v10_7.traceaad import TraceAADV107
from llm4ad.method.traceaad_v10_8 import TraceAADV108
from llm4ad.method.traceaad_v10_8.trajectory import TrajectoryBuilder, digest


def method(path, llm=None, **kwargs):
    return TraceAADV108(evaluation=TinyEvaluation(), llm=llm or FakeLLM(), run_dir=path,
                       **{'budget': 10, 'n_roots': 1, **kwargs})


def add(tree, value, parent=None, code=None, operator='Refine', donor=None):
    return tree.add(code=code or f'def score(x):\n    return {value}',
                    idea=f'RAW IDEA {len(tree.nodes)}', fitness=value,
                    evaluation_id=len(tree.nodes) + 1, parent_id=parent,
                    operator=operator if parent is not None else 'Init', donor_id=donor)


def builder(tree):
    return TrajectoryBuilder(FakeLLM(), 'TASK', lookup=tree.nodes.get,
                             max_tokens=10000, history_tokens=8192, max_events=8)


def test_allocation_arms_concentration_counts_schedule_and_resume(tmp_path):
    from llm4ad.method.traceaad_v10_3.traceaad import calibrate_beta
    runners = {arm: method(tmp_path / arm, budget=1000, allocation_arm=arm) for arm in 'ABCD'}
    for runner in runners.values():
        for score in range(100):
            add(runner.tree, score)
        runner.parent_selection_counts = {99: 99}
    a, b, c, d = [runners[arm] for arm in 'ABCD']
    beta, _, _ = calibrate_beta(list(range(100)), 0.1, 2)
    weights = [math.exp(beta * (n.fitness - 99)) / math.sqrt(1 + a.parent_selection_counts.get(n.id, 0))
               for n in a.tree.all_nodes()]
    pa, _ = a.node_distribution(a.tree.all_nodes(), 'Refine')
    assert pa == pytest.approx([w / sum(weights) for w in weights])
    pb, mb = b.node_distribution(b.tree.all_nodes(), 'Refine')
    assert mb['quality_ess'] == pytest.approx(10)
    b.parent_selection_counts = {n.id: 10000 - n.id for n in b.tree.all_nodes()}
    assert b.node_distribution(b.tree.all_nodes(), 'Refine')[0] == pb
    for used, target in [(0, 32), (500, 16), (1000, 8)]:
        d.budget_used = used
        pd, md = d.node_distribution(d.tree.all_nodes(), 'Refine')
        assert md['quality_ess'] == pytest.approx(target)
        assert md['ess_target'] == pytest.approx(target)
        pivot, _ = d.node_distribution(d.tree.all_nodes(), 'Pivot')
        assert pivot == pytest.approx([0.5*p + 0.005 for p in pd])
    pc, mc = c.node_distribution(c.tree.all_nodes(), 'Refine')
    assert mc['quality_ess'] == pytest.approx(8)
    assert pc == pytest.approx(pd)
    c._save_state()
    with pytest.raises(ValueError, match='mechanism/source/backend'):
        method(tmp_path / 'C', budget=1000, allocation_arm='B')._load_state()
    restored = method(tmp_path / 'D', budget=1000, allocation_arm='D')
    d._save_state()
    restored._load_state()
    assert restored.node_distribution(restored.tree.all_nodes(), 'Refine')[0] == pytest.approx(pd)
    tied = method(tmp_path / 'tied', allocation_arm='C')
    for _ in range(12):
        add(tied.tree, 1)
    _, mt = tied.node_distribution(tied.tree.all_nodes(), 'Refine')
    assert mt['ess_target'] == 8 and mt['attainable_ess_target'] == 12
    assert mt['quality_ess'] == pytest.approx(12)


def test_real_contiguous_edges_views_and_donor_are_traceable():
    tree = SearchTree()
    root = add(tree, 3, code='# Algorithm 72\ndef score(x):\n    return 3')
    donor = add(tree, 90)
    middle = add(tree, 2, root.id, operator='Fuse', donor=donor.id)
    current = add(tree, 4, middle.id, operator='Pivot')
    b = builder(tree)
    text, metadata = b.build(current, 'Fuse', donor)
    assert metadata['history_ids'] == [middle.id, current.id]
    assert [(r['source_id'], r['target_id'], r['source_fitness'], r['target_fitness'], r['operator'])
            for r in metadata['evidence_relations']] == [(0, 2, 3, 2, 'Fuse'), (2, 3, 2, 4, 'Pivot')]
    assert metadata['evidence_relations'][0]['historical_donor_id'] == donor.id
    assert 'additional donor participated' in text
    assert text.index('# Current') < text.index('# Recent') < text.index('# Donor') < text.index('# Design Task')
    assert text.count(current.code) == 1
    assert 'RAW IDEA' not in text and 'Algorithm 72' not in text
    assert root.code.startswith('# Algorithm 72') and root.idea == 'RAW IDEA 0'
    view = next(v for v in metadata['context_code_views'] if v['node_id'] == root.id)
    assert view['raw_code_hash'] == digest(root.code)
    assert view['view_code_hash'] == digest(b.code_view(root)[0])
    assert view['removed_reference_comments'] == 1
    hidden, _ = b.build(current, 'Pivot')
    assert '# Donor' not in hidden and 'Fitness: 90' not in hidden
    assert all(r['target_evaluation_id'] <= current.evaluation_id for r in metadata['evidence_relations'])


@pytest.mark.parametrize('trailing_newline', [False, True])
def test_complete_diff_reconstructs_both_code_views(tmp_path, trailing_newline):
    tree = SearchTree()
    lines = ['# Algorithm 99', 'def score(x):'] + [f'    a{i} = x + {i}' for i in range(100)]
    before = '\n'.join(lines) + '\n    return a99' + ('\n' if trailing_newline else '')
    after = before.replace('a1 = x + 1', 'a1 = x + 5').replace('return a99', 'return a99 + 1')
    source = add(tree, 2, code=before)
    target = add(tree, 1, source.id, code=after)
    b = builder(tree)
    block, facts = b.transition(source, target)
    assert facts['representation'] == 'diff'
    diff = block.split('```diff\n', 1)[1].rsplit('```', 1)[0]
    assert diff.count('@@') == 4  # two separated modification hunks
    assert ('\\ No newline at end of file' in diff) == (not trailing_newline)
    if shutil.which('patch') is None:
        pytest.skip('standard patch utility is required for independent reconstruction')
    path = tmp_path / 'program.py'
    path.write_text(b.code_view(source)[0])
    subprocess.run(['patch', '--batch', str(path)], input=diff, text=True, check=True, capture_output=True)
    assert path.read_text() == b.code_view(target)[0]
    subprocess.run(['patch', '--batch', '-R', str(path)], input=diff, text=True, check=True, capture_output=True)
    assert path.read_text() == b.code_view(source)[0]


def test_whole_predecessor_and_unchanged_representations():
    tree = SearchTree()
    root = add(tree, 1)
    middle = add(tree, 2, root.id)
    duplicate = add(tree, 3, middle.id, code=middle.code)
    b = builder(tree)
    block, fact = b.transition(root, middle)
    assert fact['representation'] == 'predecessor' and root.code in block
    block, fact = b.transition(middle, duplicate)
    assert fact['representation'] == 'unchanged'
    assert 'Code unchanged.' in block and 'Fitness: 2 -> 3' in block


def test_capacity_keeps_nearest_whole_suffix_and_never_skips_a_large_edge():
    tree = SearchTree()
    root = add(tree, 1)
    middle = add(tree, 2, root.id)
    current = add(tree, 3, middle.id)
    b = builder(tree)
    b.history_tokens = b.count(b.history_text([(middle, current)]))
    text, meta = b.build(current, 'Refine')
    assert meta['history_ids'] == [current.id] and meta['history_stop_reason'] == 'history_budget'
    assert b.transition(middle, current)[0] in text
    b.history_tokens -= 1
    _, meta = b.build(current, 'Refine')
    assert meta['history_ids'] == [] and meta['history_fallback_reason'] == 'history_budget'
    b.history_tokens = 8192
    minimum = b.count(b.assemble(current, 'Refine', None), chat=True)
    b.max_tokens = minimum
    text, meta = b.build(current, 'Refine')
    assert meta['history_ids'] == [] and meta['history_fallback_reason'] == 'context_budget'
    assert b.count(text, chat=True) == minimum and current.code in text
    assert b.fits(current, 'Refine')
    b.max_tokens -= 1
    with pytest.raises(ValueError, match='minimum complete'):
        b.build(current, 'Refine')


def test_root_edge_limit_and_cached_tokenizer_calls():
    tree = SearchTree()
    node = add(tree, 0)
    b = builder(tree)
    assert b.build(node, 'Refine')[1]['history_fallback_reason'] == 'root'
    for value in range(1, 11):
        node = add(tree, value, node.id)
    _, meta = b.build(node, 'Pivot')
    assert meta['history_ids'] == list(range(3, 11))
    assert meta['history_stop_reason'] == 'edge_limit'
    calls = len(b._counts)
    assert b.build(node, 'Pivot')[1] == meta and len(b._counts) == calls


def test_individual_quality_counts_and_actual_best(tmp_path):
    m = method(tmp_path)
    nodes = [add(m.tree, 1, code=f'def score(x):\n    return x + {i}') for i in range(3)]
    m.parent_selection_counts = {nodes[0].id: 8}
    refine, stats = m.node_distribution(m.eligible_nodes(), 'Refine')
    assert refine == pytest.approx([1/7, 3/7, 3/7])
    assert m.node_distribution(nodes, 'Fuse')[0] == refine
    assert m.node_distribution(nodes, 'Pivot')[0] == pytest.approx([.5*p + .5/3 for p in refine])
    assert stats['quality_ess'] == stats['attainable_ess_target'] == 3
    duplicate = add(m.tree, 1000, nodes[1].id, code=nodes[0].code)
    assert len(m.eligible_nodes()) == 4
    assert m.tree.best() is duplicate
    assert m.node_distribution(m.eligible_nodes(), 'Refine')[0][-1] > 1/4


def test_scheduler_selects_individual_and_its_own_lineage(tmp_path):
    m = method(tmp_path)
    first = add(m.tree, 1)
    other = add(m.tree, 2)
    duplicate = add(m.tree, 8, other.id, code=first.code, operator='Pivot')
    class SelectLast(random.Random):
        def choices(self, population, weights=None, **kwargs):
            return ['Refine'] if 'Refine' in population else [population[-1]]
    m.rng = SelectLast(0)
    pending = m._schedule()
    assert pending['parent_id'] == duplicate.id and pending['parent_fitness'] == 8
    assert pending['selection']['parent_route'] == 'operator_then_node'
    assert pending['evidence_relations'][0]['source_id'] == other.id
    assert pending['history_ids'] == [duplicate.id]
    assert m.parent_selection_counts == {duplicate.id: 1}


def test_operator_frequencies_and_single_fuse_fallback(tmp_path):
    m = method(tmp_path)
    add(m.tree, 1)
    counts = dict.fromkeys(['Refine', 'Pivot', 'Fuse'], 0)
    for _ in range(2000):
        p = m._schedule()
        counts[p['requested_operator']] += 1
        if p['requested_operator'] == 'Fuse':
            assert p['operator'] == 'Refine' and p['donor_id'] is None and p['parent_id'] == 0
    assert [counts[op]/2000 for op in counts] == pytest.approx([.50, .15, .35], abs=.035)
    assert m.parent_selection_counts == {0: 2000} and not m.llm.calls


def test_donor_ranked_levels_uniform_nodes_and_capacity_limit(tmp_path, monkeypatch):
    m = method(tmp_path)
    parent = add(m.tree, 0)
    low = [add(m.tree, 1, code=f'def score(x):\n    return x + {i}') for i in (1, 2)]
    high = add(m.tree, 2)
    copy = add(m.tree, 1, code=low[0].code)
    seen = dict.fromkeys([n.id for n in [*low, copy, high]], 0)
    for _ in range(2500):
        donor, attempts = m.select_donor(parent)
        seen[donor.id] += 1
        assert len(attempts) == 1
    assert [seen[n.id]/2500 for n in [*low, copy, high]] == pytest.approx([1/9, 1/9, 1/9, 2/3], abs=.04)
    for value in range(3, 45):
        add(m.tree, value)
    probes = []
    def reject(*args):
        probes.append(args[2].id)
        return False
    monkeypatch.setattr(m.builder, 'fits', reject)
    donor, attempts = m.select_donor(parent)
    assert donor is None and len(attempts) == len(probes) == 32
    assert len(set(probes)) == 32


def test_donor_capacity_has_priority_over_history():
    tree = SearchTree()
    root = add(tree, 1)
    current = add(tree, 2, root.id)
    donor = add(tree, 3)
    b = builder(tree)
    b.max_tokens = b.count(b.assemble(current, 'Fuse', donor), chat=True)
    text, meta = b.build(current, 'Fuse', donor)
    assert donor.code in text and current.code in text
    assert meta['history_ids'] == [] and meta['history_fallback_reason'] == 'context_budget'


def test_failed_evaluations_charged_but_input_copies_rejected(tmp_path):
    llm = FakeLLM(response(3), 'bad output', response('float("nan")'), response('1/0'), response(3), response(4))
    m = method(tmp_path, llm, budget=4)
    m.run()
    events = read_journal(m.events_path)
    assert len(llm.calls) == 6 and m.budget_used == 4
    assert [e['status'] for e in events] == ['ok', 'invalid_output', 'eval_failed', 'eval_failed', 'duplicate_code', 'ok']
    assert [e['budget_used'] for e in events] == [1, 1, 2, 3, 3, 4]
    assert events[4]['evaluation_id'] is None and events[4]['node_id'] is None
    assert events[4]['duplicate_matches'][0]['role'] == 'parent'
    assert sum(m.parent_selection_counts.values()) == 5
    assert len(read_journal(m.evaluations_path)) == 4 and len(m.tree.nodes) == 2
    assert m.tree.nodes[0].idea not in llm.calls[1][0]
    assert json.loads(m.state_path.read_text())['version'] == 1081
    assert m.tree.best().fitness == 4
    assert [e['best_so_far'] for e in events] == [3, 3, 3, 3, 3, 4]


@pytest.mark.parametrize('role', ['parent', 'donor'])
@pytest.mark.parametrize('view', ['raw', 'prompt'])
def test_dedup_matches_only_current_inputs_and_preserves_text_distinctions(tmp_path, role, view):
    m = method(tmp_path)
    parent = add(m.tree, 1, code='# Algorithm 72\ndef score(x):\n    return 1')
    donor = add(m.tree, 2, code='# Algorithm 73\ndef score(x):\n    return 2')
    historical = add(m.tree, 3)
    m.pending = {'parent_id': parent.id, 'donor_id': donor.id}
    node = parent if role == 'parent' else donor
    code = node.code if view == 'raw' else m.builder.code_view(node)[0]
    assert {'role': role, 'node_id': node.id, 'view': view} in m._duplicate_inputs(code.strip())
    assert not m._duplicate_inputs(code + '\n# different comment')
    assert not m._duplicate_inputs(historical.code)
    m.pending = {'parent_id': None, 'donor_id': None}
    assert not m._duplicate_inputs(parent.code)


@pytest.mark.parametrize('crash_point', ['after_event', 'after_checkpoint'])
def test_duplicate_recovery_does_not_regenerate_evaluate_or_recount(tmp_path, monkeypatch, crash_point):
    m = method(tmp_path, FakeLLM(response(1), response(1)))
    m._advance()
    original = m._append_record if crash_point == 'after_event' else m._save_state
    def crash(*args):
        original(*args)
        if crash_point == 'after_checkpoint' or args[0] == m.events_path:
            raise RuntimeError('simulated crash')
    monkeypatch.setattr(m, '_append_record' if crash_point == 'after_event' else '_save_state', crash)
    with pytest.raises(RuntimeError, match='simulated crash'):
        m._advance()
    resumed = method(tmp_path, FakeLLM(response(2)))
    resumed._load_state()
    if resumed.pending is not None:
        resumed._advance()
    assert resumed.budget_used == 1 and resumed.completed_attempts == 2
    assert resumed.parent_selection_counts == {0: 1}
    assert len(read_journal(resumed.events_path)) == 2
    assert len(read_journal(resumed.evaluations_path)) == 1
    assert not resumed.llm.calls
    resumed._advance()
    assert resumed.budget_used == 2 and resumed.completed_attempts == 3


@pytest.mark.parametrize('view', ['raw', 'prompt'])
def test_fuse_donor_copy_rejected_before_evaluator(tmp_path, monkeypatch, view):
    m = method(tmp_path)
    parent = add(m.tree, 1)
    donor = add(m.tree, 2, code='# Algorithm 72\ndef score(x):\n    return 2')
    class FuseFirst(random.Random):
        def choices(self, population, weights=None, **kwargs):
            return ['Fuse'] if 'Fuse' in population else [population[0]]
    m.rng = FuseFirst(0)
    code = donor.code if view == 'raw' else m.builder.code_view(donor)[0]
    m.llm = FakeLLM('Idea: Copy donor.\n```python\n' + code + '\n```')
    def fail_if_evaluated(*args):
        pytest.fail('duplicate must never reach evaluator')
    monkeypatch.setattr(m, '_evaluate_pending', fail_if_evaluated)
    m._advance()
    event = read_journal(m.events_path)[0]
    assert event['operator'] == 'Fuse' and event['donor_id'] == donor.id
    assert event['status'] == 'duplicate_code'
    assert all(match['role'] == 'donor' for match in event['duplicate_matches'])
    assert m.budget_used == 0 and m.completed_attempts == 1 and len(m.tree.nodes) == 2
    assert m.parent_selection_counts == {parent.id: 1}


def test_fifty_input_copies_stop_without_spending_evaluation_budget(tmp_path):
    m = method(tmp_path, FakeLLM(*[response(1)] * 51))
    with pytest.raises(RuntimeError, match='50 consecutive'):
        m.run()
    assert m.budget_used == 1 and m.completed_attempts == 51
    assert m.parent_selection_counts == {0: 50}
    assert len(m.tree.nodes) == len(read_journal(m.evaluations_path)) == 1
    assert json.loads(m.summary_path.read_text())['status'] == 'error'


def test_old_v108_checkpoint_rejected_without_writing(tmp_path):
    m = method(tmp_path, FakeLLM(response(1)), budget=1)
    m.run()
    state = json.loads(m.state_path.read_text())
    state['version'] = 1080
    m.state_path.write_text(json.dumps(state))
    before = m.summary_path.read_bytes()
    with pytest.raises(ValueError, match='configuration'):
        method(tmp_path, budget=1).run()
    assert m.summary_path.read_bytes() == before


@pytest.mark.parametrize('name', [
    'test_transport_retry_restores_same_parent_rng_and_selection_count',
    'test_persisted_response_survives_crash_before_evaluator',
    'test_unknown_evaluation_blocks_instead_of_redrawing_or_refunding',
])
def test_inherited_execution_contract(tmp_path, monkeypatch, name):
    monkeypatch.setattr(execution_contract, 'method', method)
    args = (tmp_path,) if 'transport' in name else (tmp_path, monkeypatch)
    getattr(execution_contract, name)(*args)


@pytest.mark.parametrize('crash_point', ['after_receipt', 'after_event', 'after_checkpoint'])
def test_idempotent_crash_recovery(tmp_path, monkeypatch, crash_point):
    monkeypatch.setattr(execution_contract, 'method', method)
    execution_contract.test_receipts_and_candidate_commits_are_idempotent(tmp_path, monkeypatch, crash_point)


def test_foreign_checkpoint_and_configuration_drift_do_not_overwrite_summary(tmp_path):
    old = TraceAADV107(evaluation=TinyEvaluation(), llm=FakeLLM(response()), run_dir=tmp_path,
                      budget=1, n_roots=1)
    old.run()
    before = old.summary_path.read_bytes()
    with pytest.raises(ValueError, match='V10.8 configuration'):
        method(tmp_path, budget=1).run()
    assert old.summary_path.read_bytes() == before
    path = tmp_path / 'v108'
    m = method(path, FakeLLM(response()), budget=1)
    m.run()
    before = m.summary_path.read_bytes()
    for kwargs in ({'history_tokens': 20}, {'seed': 3}, {'traj_gens': 1}):
        with pytest.raises(ValueError, match='configuration'):
            method(path, budget=1, **kwargs).run()
        assert m.summary_path.read_bytes() == before
    changed = method(path, budget=1)
    changed.mechanism['evaluation_config']['timeout_seconds'] = 9
    with pytest.raises(ValueError, match='configuration'):
        changed.run()


def test_independent_ablation_constructor_and_production_defaults(tmp_path):
    from experiments.traceaad_v10_8.ablations import build_history_ablation
    from experiments.traceaad_v10_8.run import build_parser
    for arm, edges in [('code_only', 0), ('single_edge', 1), ('multi_edge', 8)]:
        m = build_history_ablation(arm, evaluation=TinyEvaluation(), llm=FakeLLM(), run_dir=tmp_path / arm)
        assert m.builder.max_events == edges
        assert m.mechanism['experiment_arm'] == arm
        assert m.builder.max_tokens == 16128
    args = build_parser().parse_args(['--task', 'tsp_construct'])
    assert args.n_roots == 8 and args.budget == 1000
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--task', 'tsp_construct', '--history-mode', 'old'])


def test_full_suffix_counts_only_final_chat_and_cached_donor_change():
    tree = SearchTree()
    node = add(tree, 0)
    for value in range(1, 9):
        node = add(tree, value, node.id)
    donor = add(tree, 20)
    b = builder(tree)
    b.build(node, 'Refine')
    assert sum(chat for chat, _ in b._counts) == 1
    # 16 representation comparisons, one full history count, one chat count.
    assert len(b._counts) == 18
    before = set(b._counts)
    b.build(node, 'Fuse', donor)
    assert len(set(b._counts) - before) == 1
    assert sum(chat for chat, _ in set(b._counts) - before) == 1


def test_semantic_pair_keeps_same_snapshot_operator_donor_and_visible_edges(tmp_path):
    from experiments.traceaad_v10_8.ablations import build_representation_pair
    m = method(tmp_path)
    root = add(m.tree, 1)
    middle = add(m.tree, 2, root.id)
    current = add(m.tree, 3, middle.id)
    donor = add(m.tree, 4)
    pair = build_representation_pair(m, parent_id=current.id, operator='Fuse', donor_id=donor.id)
    code, idea = pair['code_transitions'], pair['short_idea']
    assert code['history_ids'] == idea['history_ids'] == [middle.id, current.id]
    assert current.idea in idea['prompt'] and current.idea not in code['prompt']
    for item in pair.values():
        assert item['parent_id'] == current.id and item['donor_id'] == donor.id
        assert item['max_input_tokens'] == 16128
        assert item['snapshot_evaluation_id'] == donor.evaluation_id
        assert item['prompt_tokens'] == m.llm.count_prompt_tokens(item['prompt'])
    m.builder.history_tokens = 1
    pair = build_representation_pair(m, parent_id=current.id, operator='Refine')
    assert all(item['history_ids'] == [] for item in pair.values())
    assert not m.llm.calls and m.completed_attempts == 0

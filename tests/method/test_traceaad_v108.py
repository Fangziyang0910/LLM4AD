import json
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


def test_duplicate_opportunity_conservation_first_quality_and_group_count(tmp_path):
    m = method(tmp_path)
    originals = [add(m.tree, 1, code=f'def score(x):\n    return x + {i}') for i in range(3)]
    p, _ = m.group_distribution(m.eligible_groups(), 'Refine')
    duplicate = add(m.tree, 1000, originals[1].id, code=originals[0].code)
    groups = m.eligible_groups()
    assert m.group_distribution(groups, 'Refine')[0] == pytest.approx(p)
    assert groups[0][0] is originals[0] and groups[0][1] == [originals[0], duplicate]
    m.parent_selection_counts = {originals[0].id: 3, duplicate.id: 5}
    groups = m.eligible_groups()
    refine, stats = m.group_distribution(groups, 'Refine')
    assert refine == pytest.approx([1/7, 3/7, 3/7])
    assert m.group_distribution(groups, 'Fuse')[0] == refine
    assert m.group_distribution(groups, 'Pivot')[0] == pytest.approx([.5*p + .5/3 for p in refine])
    assert stats['ess_target'] == 2 and stats['quality_ess'] == 3
    assert stats['attainable_ess_target'] == 3
    assert m.tree.best() is originals[0]


def test_scheduler_returns_whole_duplicate_record_and_its_own_lineage(tmp_path):
    m = method(tmp_path)
    first = add(m.tree, 1)
    other = add(m.tree, 2)
    duplicate = add(m.tree, 8, other.id, code=first.code, operator='Pivot')
    # Deterministically select the first group, then its last real record.
    class SelectLast(random.Random):
        def choices(self, population, weights=None, **kwargs):
            return ['Refine'] if 'Refine' in population else [0]
        def choice(self, population):
            return population[-1]
    m.rng = SelectLast(0)
    pending = m._schedule()
    assert pending['parent_id'] == duplicate.id and pending['parent_fitness'] == 8
    assert pending['selection']['group_fitness'] == 1
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


def test_donor_ranked_levels_uniform_implementations_and_capacity_limit(tmp_path, monkeypatch):
    m = method(tmp_path)
    parent = add(m.tree, 0)
    low = [add(m.tree, 1, code=f'def score(x):\n    return x + {i}') for i in (1, 2)]
    high = add(m.tree, 2)
    add(m.tree, 999, code=low[0].code)
    seen = dict.fromkeys([n.code for n in [*low, high]], 0)
    for _ in range(2500):
        donor, attempts = m.select_donor(parent)
        seen[donor.code] += 1
        assert len(attempts) == 1
    assert [seen[n.code]/2500 for n in [*low, high]] == pytest.approx([1/6, 1/6, 2/3], abs=.04)
    for value in range(3, 45):
        add(m.tree, value)
    probes = []
    def reject(*args):
        probes.append(args[2].id)
        return False
    monkeypatch.setattr(m.builder, 'fits', reject)
    donor, attempts = m.select_donor(parent)
    assert donor is None and len(attempts) == len(probes) == 32
    assert len({m.tree.nodes[i].code for i in probes}) == 32


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


def test_failed_and_duplicate_evaluations_are_charged_and_ideas_preserved(tmp_path):
    llm = FakeLLM(response(3), 'bad output', response('float("nan")'), response('1/0'), response(3))
    m = method(tmp_path, llm, budget=4)
    m.run()
    events = read_journal(m.events_path)
    assert len(llm.calls) == 5 and m.budget_used == 4
    assert [e['status'] for e in events] == ['ok', 'invalid_output', 'eval_failed', 'eval_failed', 'ok']
    assert [e['budget_used'] for e in events] == [1, 1, 2, 3, 4]
    assert sum(m.parent_selection_counts.values()) == 4
    assert len(read_journal(m.evaluations_path)) == 4
    assert len(m.tree.nodes) == 2 and m.tree.nodes[0].code == m.tree.nodes[1].code
    assert m.tree.nodes[0].idea == m.tree.nodes[1].idea
    assert m.tree.nodes[0].idea not in llm.calls[1][0]
    state = json.loads(m.state_path.read_text())
    assert state['version'] == 1080 and state['mechanism']['history_tokens'] == 8192


def test_final_best_uses_first_score_even_when_record_no_longer_fits(tmp_path, monkeypatch):
    llm = FakeLLM(response(1), response(2), response(1))
    m = method(tmp_path, llm, budget=3)
    from llm4ad.base.evaluate import EvaluationOutcome
    scores = iter([1, 2, 100])
    monkeypatch.setattr(m.secure, 'evaluate_program_with_details', lambda code: EvaluationOutcome(next(scores)))
    m.run()
    summary = json.loads(m.summary_path.read_text())
    assert summary['best']['fitness'] == 2
    assert summary['fitness_instability'][0]['first_fitness'] == 1
    assert [e['best_so_far'] for e in read_journal(m.events_path)] == [1, 2, 2]
    assert read_journal(m.events_path)[-1]['implementation_fitness'] == 1
    m.builder.max_tokens = 1
    assert m.tree.best().fitness == 2


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

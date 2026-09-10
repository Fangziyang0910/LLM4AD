"""Independent constructors; all arms share the production individual scheduler and input deduplication."""

from pathlib import Path
from dataclasses import replace

from llm4ad.method.traceaad_v10_8 import TraceAADV108
from llm4ad.method.traceaad_v10_3.schema import SearchTree
from llm4ad.method.traceaad_v10_8.trajectory import TrajectoryBuilder, digest


def build_history_ablation(arm, **kwargs):
    limits = {'code_only': 0, 'single_edge': 1, 'multi_edge': 8}
    runner = TraceAADV108(traj_gens=limits[arm], **kwargs)
    runner.mechanism['experiment_arm'] = arm
    runner.mechanism['source_hashes'][str(Path(__file__).resolve())] = digest(Path(__file__).read_text())
    return runner


class SemanticTrajectoryBuilder(TrajectoryBuilder):
    """Short original Idea/Result/Fitness evidence, only for anchor experiments."""

    def transition(self, source, target):
        if target.parent_id != source.id:
            raise ValueError('formation evidence must be a real direct edge')
        key = (source.id, target.id)
        if key not in self._transitions:
            idea, omitted = self.idea_view(target)
            result = ('improved' if target.fitness > source.fitness else
                      'regressed' if target.fitness < source.fitness else 'unchanged')
            text = (f'Executed operator: {target.operator}\n'
                    f'Fitness: {source.fitness} -> {target.fitness}\n'
                    f'Result: {result}\nIdea: {idea or "[omitted]"}')
            if target.donor_id is not None:
                text += '\nAn additional donor participated in this multi-input transition.'
            self._transitions[key] = text, {
                'source_id': source.id, 'target_id': target.id,
                'source_evaluation_id': source.evaluation_id,
                'target_evaluation_id': target.evaluation_id,
                'source_fitness': source.fitness, 'target_fitness': target.fitness,
                'operator': target.operator, 'historical_donor_id': target.donor_id,
                'representation': 'short_idea', 'idea_omission_reason': omitted,
                'text_hash': digest(text),
            }
        return self._transitions[key]

    def history_text(self, edges):
        return ('# Recent Formation Transitions\n'
                'These are consecutive actual transitions, oldest first, ending at '
                'the current implementation. Original Ideas are unverified descriptions; '
                'fitness describes the whole measured transition.\n\n' +
                '\n\n'.join(self.transition(*edge)[0] for edge in edges))


def build_representation_pair(runner, *, parent_id, operator, donor_id=None):
    """Freeze one visible archive and render matched-history prompts; no LLM draws.

    The experiment caller samples each returned prompt with the same generation
    parameters and evaluates under the same protocol. Persist both texts and
    metadata before drawing; invalid outputs remain in the trial denominator.
    """
    if operator not in ('Refine', 'Pivot', 'Fuse') or (operator == 'Fuse') != (donor_id is not None):
        raise ValueError('paired anchor must specify a valid operator and Fuse donor')
    snapshot = SearchTree()
    for node in runner.tree.all_nodes():
        snapshot.add_raw(replace(node))
    parent = snapshot.nodes[parent_id]
    donor = snapshot.nodes[donor_id] if donor_id is not None else None
    if donor is not None and donor.id == parent.id:
        raise ValueError('Fuse donor must be a different node')
    builders = {
        name: cls(runner.llm, runner.task_contract, lookup=snapshot.nodes.get,
                  max_tokens=runner.builder.max_tokens,
                  history_tokens=runner.builder.history_tokens, max_events=runner.builder.max_events,
                  log_count=runner.builder.log_count)
        for name, cls in [('code_transitions', TrajectoryBuilder), ('short_idea', SemanticTrajectoryBuilder)]
    }
    results = {name: b.build(parent, operator, donor) for name, b in builders.items()}
    # Different representation costs must not silently change the compared
    # history. Restrict both to their common nearest suffix, with no padding.
    depth = min(meta['history_edge_count'] for _, meta in results.values())
    for name, b in builders.items():
        if results[name][1]['history_edge_count'] != depth:
            b.max_events = depth
            results[name] = b.build(parent, operator, donor)
    return {
        name: {'prompt': text, 'prompt_hash': digest(text),
               'prompt_tokens': builders[name].count(text, chat=True),
               'parent_id': parent_id, 'donor_id': donor_id, 'operator': operator,
               'snapshot_evaluation_id': max(n.evaluation_id for n in snapshot.all_nodes()),
               'max_input_tokens': runner.builder.max_tokens, **meta}
        for name, (text, meta) in results.items()
    }

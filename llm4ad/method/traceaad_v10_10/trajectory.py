"""Complete formation history, conditioned on the requested design task."""

from difflib import unified_diff
from llm4ad.method.traceaad_v10_8.trajectory import TrajectoryBuilder as BaseBuilder, digest
from .errors import OUTPUT

GENERATION = 'idea_code_tolerant_one_repair_v1'

CONTEXT_POLICY = 'operator_context_single_capacity_check_v3'
INITIALIZATION_POLICY = 'sequential_informed_v1'
INSTRUCTIONS = {
    'Init': 'Design a competitive coherent decision method with meaningful task-dependent computations. '
            'When previous initial algorithms are shown, study their decision rules and evaluated '
            'performance, then design another competitive candidate algorithm.',
    'Refine': 'Continue improving the current algorithm within its existing design. '
              'Make one focused modification based on the current code and its formation history.',
    'Tune': 'Identify the main algorithm parameters and improve their settings while preserving '
            'the core decision method.',
    'Pivot': 'Use the current algorithm as a reference to design a competitive, materially different '
             'decision method. You may replace the core design or reorganize useful computations; '
             'do not merely tune parameters.',
    'Fuse': 'Use the current implementation as the host and draw on useful computations from the donor. '
            'Adapt, replace or reorganize computations as needed to improve the host. '
            'If the donor has no useful computation, improve the host without forced mixing. '
            'Aim to outperform both inputs.',
}
NOTE = ('# Formation History\nEach block is an actual source-to-target formation edge, not a '
        'chronological chain between blocks. Fitness measures the whole program (higher is better); '
        'it does not identify the causal contribution of a component. No reported history means no '
        'evidence, not that an idea failed. Changes must affect computations actually consumed by '
        'the evaluator, rather than masked entries or unused terms.')
TEMPLATE_HASH = digest(str(INSTRUCTIONS) + NOTE + OUTPUT)


class TrajectoryBuilder(BaseBuilder):
    def __init__(self, *args, all_nodes, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_nodes = all_nodes

    def assemble(self, parent, operator, donor, blocks=()):
        parts = [self.task_contract]
        if parent is not None:
            parts.append(self.program(parent, 'Reference Algorithm' if operator == 'Pivot'
                                      else 'Current Implementation'))
        if donor is not None:
            parts.append(self.program(donor, 'Donor'))
        if blocks:
            parts.append(NOTE + '\n\n' + '\n\n'.join(blocks))
        parts.extend(['# Design Task\n' + INSTRUCTIONS[operator], '# Output\n' + OUTPUT])
        return '\n\n\n'.join(parts)

    def program(self, node, title):
        return super().program(node, f'{title} (node {node.id})')

    def trial(self, source, target, role):
        # Keep each formation edge independently reconstructable.
        if target.parent_id != source.id:
            raise ValueError('trial must be a real direct edge')
        before, after = self.code_view(source)[0], self.code_view(target)[0]
        patch = ''.join(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n'
                        for line in unified_diff(before.splitlines(True), after.splitlines(True),
                                                 fromfile='source.py', tofile='target.py'))
        diff = f'Complete source-to-target diff:\n```diff\n{patch}```'
        # Formation edges include their complete source code.
        text = (f'## {role}: node {source.id} -> node {target.id}\n'
                f'Evaluation: {source.evaluation_id} -> {target.evaluation_id}\n'
                f'Executed operator: {target.operator}; Fitness: {source.fitness} -> {target.fitness}\n')
        if target.donor_id is not None:
            text += f'Historical donor node {target.donor_id} also participated; this block does not isolate its effect.\n'
        text += f'Complete source code:\n```python\n{before}\n```\n'
        text += diff
        if target.donor_id is not None:
            historical_donor = self.lookup(target.donor_id)
            text += '\n' + self.program(historical_donor, 'Historical transfer source')
        return text, {'source_id': source.id, 'target_id': target.id, 'role': role,
                      'source_fitness': source.fitness, 'target_fitness': target.fitness,
                      'source_evaluation_id': source.evaluation_id,
                      'target_evaluation_id': target.evaluation_id,
                      'historical_donor_id': target.donor_id, 'operator': target.operator,
                      'text_hash': digest(text)}

    def evidence(self, parent, operator, donor):
        if parent is None or operator in ('Tune', 'Pivot'):
            return []
        edges = []
        if operator == 'Fuse' and donor is not None and donor.parent_id is not None:
            edges.append((self.lookup(donor.parent_id), donor, 'Donor formation'))
        cursor = parent
        for _ in range(min(3, self.max_events)):
            if cursor.parent_id is None:
                break
            source = self.lookup(cursor.parent_id)
            edges.append((source, cursor, 'Host formation'))
            cursor = source
        return edges

    def build_initial(self):
        # Sequential informed initialization: the first root sees only the task;
        # each later root sees every previously evaluated root, code and fitness.
        roots = sorted((n for n in self.all_nodes() if n.parent_id is None), key=lambda n: n.id)
        parts = [self.task_contract]
        if roots:
            parts.append('# Previous Initial Algorithms\n'
                         'Complete previously evaluated programs, in generation order. '
                         'Fitness: higher is better.')
            parts.extend(self.program(n, 'Previous initial algorithm') for n in roots)
        parts.extend(['# Design Task\n' + INSTRUCTIONS['Init'], '# Output\n' + OUTPUT])
        text = '\n\n\n'.join(parts)
        self.check_capacity(text)
        return text, {}

    def build(self, parent, operator, donor=None):
        if operator == 'Init':
            return self.build_initial()
        blocks, relations, shown = [], [], {}
        for node in (parent, donor):
            if node is not None:
                shown[node.id] = node
        for source, target, role in self.evidence(parent, operator, donor):
            block, facts = self.trial(source, target, role)
            blocks.append(block)
            relations.append(facts)
            shown.update({source.id: source, target.id: target})
            if target.donor_id is not None:
                shown[target.donor_id] = self.lookup(target.donor_id)
        text = self.assemble(parent, operator, donor, blocks)
        self.check_capacity(text)
        return text, {
            'history_ids': [r['target_id'] for r in relations],
            'history_edge_count': len(relations),
            'evidence_relations': relations,
            'context_node_ids': list(shown),
            'context_code_views': [self.view_record(n) for n in shown.values()],
            'context_best_fitness': max((n.fitness for n in shown.values()), default=None),
        }

    def check_capacity(self, text):
        if self.count(text, chat=True) > self.max_tokens:
            raise ValueError('complete prompt exceeds the model context budget')

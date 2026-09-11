"""Unified short formation path for normal search generation."""

from llm4ad.method.traceaad_v10_8.trajectory import TrajectoryBuilder as BaseBuilder, digest
from .errors import OUTPUT

GENERATION = 'code_first_one_repair_v1'

CONTEXT_POLICY = 'unified_short_formation_path_v1'
INITIALIZATION_POLICY = 'sequential_informed_v1'
INSTRUCTIONS = {
    'Init': 'Design a competitive candidate algorithm for the task, using the previously '
            'evaluated initial algorithms and their fitness as context when available.',
    'Refine': 'Improve the current algorithm within its existing design.',
    'Tune': 'Improve the parameter settings while preserving the current decision method.',
    'Pivot': 'Develop a competitive alternative decision method.',
    'Fuse': 'Improve the current algorithm by drawing on useful computations from the donor. '
            'Aim to outperform both inputs; if the donor has no useful computation, '
            'improve the host without forced mixing.',
}
HISTORY_TITLE = '# Formation History — oldest to newest'
HISTORY_NOTE = 'Use the available formation history to guide the next design.'
TEMPLATE_HASH = digest(str(INSTRUCTIONS) + HISTORY_TITLE + HISTORY_NOTE + OUTPUT)


class TrajectoryBuilder(BaseBuilder):
    def __init__(self, *args, all_nodes, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_nodes = all_nodes

    def formation_edges(self, current):
        """Recent formation edges of the current node, oldest first."""
        edges = []
        for _ in range(self.max_events):
            if current is None or current.parent_id is None:
                break
            source = self.lookup(current.parent_id)
            if source is None:
                raise ValueError('missing archived formation predecessor')
            edges.append((source, current))
            current = source
        return list(reversed(edges))

    def render_history(self, edges):
        blocks = []
        for step, (source, target) in enumerate(edges, start=1):
            lines = [f'Step {step} | {target.operator} | '
                     f'Fitness: {source.fitness} -> {target.fitness}']
            if target.idea:
                # A step without a description keeps its operator and scores;
                # no description is invented for it.
                lines.append('Idea: ' + ' '.join(target.idea.split()))
            blocks.append('\n'.join(lines))
        return '\n\n'.join(blocks)

    def program(self, node, title):
        return f'# {title}\nFitness: {node.fitness}\n```python\n{node.code}\n```'

    def assemble(self, parent, operator, donor, history=''):
        parts = [self.task_contract]
        if parent is not None:
            parts.append(self.program(parent, 'Current Algorithm'))
        if donor is not None:
            parts.append(self.program(donor, 'Donor'))
        if history:
            parts.append(f'{HISTORY_TITLE}\n{HISTORY_NOTE}\n\n{history}')
        parts.extend(['# Design Task\n' + INSTRUCTIONS[operator], '# Output\n' + OUTPUT])
        return '\n\n\n'.join(parts)

    def build_initial(self):
        # Sequential informed initialization: the first root sees only the task;
        # each later root sees every previously evaluated root, raw code and fitness.
        roots = sorted((n for n in self.all_nodes() if n.parent_id is None), key=lambda n: n.id)
        parts = [self.task_contract]
        if roots:
            parts.append('# Previous Initial Algorithms\n'
                         'Complete previously evaluated programs, in generation order. '
                         'Fitness: higher is better.')
            parts.extend(f'# Previous Initial Algorithm\n'
                         f'Fitness: {n.fitness}\n'
                         f'```python\n{n.code}\n```' for n in roots)
        parts.extend(['# Design Task\n' + INSTRUCTIONS['Init'], '# Output\n' + OUTPUT])
        text = '\n\n\n'.join(parts)
        self.check_capacity(text)
        return text, {}

    def build(self, parent, operator, donor=None):
        if operator == 'Init':
            return self.build_initial()
        edges = self.formation_edges(parent)
        text = self.assemble(parent, operator, donor, self.render_history(edges))
        self.check_capacity(text)
        scores = [n.fitness for n in (parent, donor) if n is not None]
        scores += [fitness for s, t in edges for fitness in (s.fitness, t.fitness)]
        return text, {
            'history_edges': [[s.id, t.id] for s, t in edges],
            'context_node_ids': [n.id for n in (parent, donor) if n is not None],
            'context_best_fitness': max(scores, default=None),
        }

    def check_capacity(self, text):
        if self.count(text, chat=True) > self.max_tokens:
            raise ValueError('complete prompt exceeds the model context budget')

"""V10.6 task, operator, and history prompts with one Idea-and-Code output."""

from dataclasses import dataclass
import hashlib

from llm4ad.method.traceaad_v10_5.prompts import PromptBuilder as BaseBuilder, formation_events
from llm4ad.method.traceaad_v10_6.prompts import (
    HISTORY_GUIDANCE, INSTRUCTIONS, OUTPUT, build_task_contract,
)

GENERATION = 'idea_code_single_call'
TEMPLATE_HASH = hashlib.sha256(
    (HISTORY_GUIDANCE + str(INSTRUCTIONS) + OUTPUT).encode()
).hexdigest()


@dataclass(frozen=True)
class Prompt:
    text: str
    tokens: int
    history_ids: tuple[int, ...]
    omissions: tuple[str, ...]
    history_tokens: int


class PromptBuilder(BaseBuilder):
    def __init__(self, *args, lookup=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookup = lookup or (lambda _: None)

    def render_history(self, events):
        lines = ['# Development History', HISTORY_GUIDANCE]
        for index, (child, parent) in enumerate(events, 1):
            donor = self.lookup(child.donor_id) if child.donor_id is not None else None
            text = f'Step {index}\nPrevious version fitness: {parent.fitness}\n'
            if donor is not None:
                text += f'Reference algorithm fitness: {donor.fitness}\n'
            text += f'Resulting version fitness: {child.fitness}\nIdea: {child.idea}'
            lines.append(text)
        return '\n\n'.join(lines)

    def assemble(self, current, operator, donor, events, omitted=None):
        omitted = omitted or set()
        parts = [self.task_contract]
        for title, node, show_idea in [
            ('Current Algorithm', current, current is not None and current.parent_id is None),
            ('Reference Algorithm', donor, True),
        ]:
            if node is None:
                continue
            text = f'# {title}\nFitness: {node.fitness}\n\n```python\n{self._code(node)}\n```'
            if show_idea and node.id not in omitted:
                text += '\nIdea: ' + node.idea
            parts.append(text)
        if events:
            parts.append(self.render_history(events))
        parts.append('# Algorithm Design Task\n' + INSTRUCTIONS[operator])
        parts.append('# Output\n' + OUTPUT)
        return '\n\n\n'.join(parts)

    def fits(self, current, operator, donor=None):
        omitted = {node.id for node in [current, donor] if node is not None}
        return self.count(
            self.assemble(current, operator, donor, [], omitted), chat=True
        ) <= self.max_tokens

    def build(self, current, ancestors, operator, donor=None):
        events = formation_events(current, ancestors, self.max_events)
        reasons, omitted = [], set()
        while events and self.count(self.render_history(events)) > self.history_tokens:
            reasons.append(f'history_budget:{events.pop(0)[0].id}')
        while True:
            text = self.assemble(current, operator, donor, events, omitted)
            tokens = self.count(text, chat=True)
            if tokens <= self.max_tokens:
                break
            if events:
                reasons.append(f'context_budget:{events.pop(0)[0].id}')
                continue
            candidate = next(
                (node for node in [donor, current]
                 if node is not None and node.id not in omitted
                 and (node is donor or node.parent_id is None)),
                None,
            )
            if candidate is None:
                raise ValueError('minimum complete prompt exceeds the model context budget')
            omitted.add(candidate.id)
            reasons.append(f'context_idea:{candidate.id}')
        return Prompt(
            text, tokens, tuple(node.id for node, _ in events), tuple(reasons),
            self.count(self.render_history(events)) if events else 0,
        )


TRAJECTORY_INSTRUCTIONS = {
    'Init': INSTRUCTIONS['Init'],
    'Refine': "Build on Algorithm {parent}'s main idea to design an improved version. "
              'Compare the supplied implementations and adapt useful parts into a coherent algorithm.',
    'Pivot': 'Use the supplied algorithms and their evaluation results to design an alternative '
             'main decision method for this task. Select useful parts and implement a complete algorithm.',
    'Fuse': 'Combine Algorithm {parent} and Algorithm {donor} as the main inputs. '
            'Use the other supplied algorithms as additional references. Select and adapt compatible '
            'computations into a coherent algorithm, retaining, replacing or reorganizing parts as useful.',
}
TRAJECTORY_OUTPUT = OUTPUT + '\nKeep the Idea within 100 words.'
IDEA_TOKENS = 256
TRAJECTORY_TEMPLATE_HASH = hashlib.sha256(
    (str(TRAJECTORY_INSTRUCTIONS) + TRAJECTORY_OUTPUT + str(IDEA_TOKENS)).encode()
).hexdigest()


class TrajectoryBuilder(PromptBuilder):
    def idea_view(self, node):
        if not hasattr(self, '_idea_views'):
            self._idea_views = {}
        if node.id not in self._idea_views:
            self._idea_views[node.id] = (node.idea if self.count(node.idea) <= IDEA_TOKENS else '')
        return self._idea_views[node.id]

    def program_text(self, node, index, omit_idea=False):
        idea = '' if omit_idea else self.idea_view(node)
        return (f'Algorithm {index}\nFitness: {node.fitness}\nIdea: {idea}\n'
                f'Code:\n```python\n{node.code}\n```')

    def trajectory(self, parent, references, operator):
        nodes = sorted(([parent] if parent is not None else []) + list(references),
                       key=lambda node: (node.fitness, node.id))
        donor = max(references, key=lambda node: (node.fitness, -node.id)) if references and operator == 'Fuse' else None
        executed = 'Refine' if operator == 'Fuse' and donor is None else operator
        positions = {node.id: index for index, node in enumerate(nodes, 1)}
        instruction = TRAJECTORY_INSTRUCTIONS[executed].format(
            parent=positions.get(parent.id) if parent else None,
            donor=positions.get(donor.id) if donor else None,
        )
        blocks = [self.program_text(node, index) for index, node in enumerate(nodes, 1)]
        def assemble():
            parts = [self.task_contract]
            if blocks:
                parts.append('# Algorithm Trajectory\n\n' + '\n\n'.join(blocks))
            parts.extend(['# Algorithm Design Task\n' + instruction, '# Output\n' + TRAJECTORY_OUTPUT])
            return '\n\n\n'.join(parts)
        text = assemble()
        if self.count(text, chat=True) > self.max_tokens:
            # Auxiliary design prose must not exclude otherwise fitting code.
            blocks = [self.program_text(node, index, omit_idea=True)
                      for index, node in enumerate(nodes, 1)]
            text = assemble()
        return text, nodes, donor, executed, blocks

    def fits_references(self, parent, references, operator):
        text, *_ = self.trajectory(parent, references, operator)
        return self.count(text, chat=True) <= self.max_tokens

    def fits(self, current, operator, donor=None):
        return self.fits_references(current, [donor] if donor else [], operator)

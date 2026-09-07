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

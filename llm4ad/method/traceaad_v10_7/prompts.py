"""V10.7 task-specific evidence prompts with one Idea-and-Code output."""

import hashlib
import io
import re
import tokenize

from llm4ad.method.traceaad_v10_5.prompts import PromptBuilder as BaseBuilder
from llm4ad.method.traceaad_v10_6.prompts import (
    INSTRUCTIONS, OUTPUT as BASE_OUTPUT, build_task_contract,
)

GENERATION = 'idea_code_single_call_self_contained_v1'
TEMPORARY_REFERENCE_RE = re.compile(r'(?i)\bAlgorithm\s+\d+\b|算法\s*\d+')
OUTPUT = BASE_OUTPUT + (
    '\nThe Idea must stand on its own: state this algorithm\'s main decision rule and key '
    'computation without referring to Algorithm numbers or temporary display positions.'
)


TRAJECTORY_INSTRUCTIONS = {
    'Init': INSTRUCTIONS['Init'],
    'Refine': "Use Algorithm {parent} as the design base and test one main improvement hypothesis. "
              'Use the supplied contrast to judge what should change, while preserving unrelated '
              'effective computation unless the hypothesis requires a supporting change.',
    'Pivot': 'Use Algorithm {parent} as the comparison baseline, but do not inherit it by default. '
             'Design a competitive alternative main decision method, making a substantive decision '
             'change rather than a cosmetic formula rewrite.',
    'Fuse': 'Use Algorithm {parent} as the design base and Algorithm {donor} as the transfer source. '
            'Adapt compatible computation into one coherent algorithm and aim to improve on the best '
            'input. Other supplied material is contrast evidence, not another required parent.',
}
TRAJECTORY_OUTPUT = OUTPUT + '\nKeep the Idea within 100 words.'
IDEA_TOKENS = 256
TRAJECTORY_TEMPLATE_HASH = hashlib.sha256(
    (str(TRAJECTORY_INSTRUCTIONS) + TRAJECTORY_OUTPUT + str(IDEA_TOKENS)).encode()
).hexdigest()


class TrajectoryBuilder(BaseBuilder):
    def idea_view(self, node):
        if not hasattr(self, '_idea_views'):
            self._idea_views = {}
        if node.id not in self._idea_views:
            if TEMPORARY_REFERENCE_RE.search(node.idea):
                self._idea_views[node.id] = ('', 'temporary_algorithm_reference')
            elif self.count(node.idea) > IDEA_TOKENS:
                self._idea_views[node.id] = ('', 'idea_token_limit')
            else:
                self._idea_views[node.id] = (node.idea, None)
        return self._idea_views[node.id]

    def code_view(self, node):
        if not hasattr(self, '_code_views'):
            self._code_views = {}
        if node.id not in self._code_views:
            removed = 0
            try:
                tokens = []
                for token in tokenize.generate_tokens(io.StringIO(node.code).readline):
                    if token.type == tokenize.COMMENT and TEMPORARY_REFERENCE_RE.search(token.string):
                        token = tokenize.TokenInfo(token.type, '', token.start, token.end, token.line)
                        removed += 1
                    tokens.append(token)
                view = tokenize.untokenize(tokens)
            except (IndentationError, tokenize.TokenError):
                view, removed = node.code, 0
            self._code_views[node.id] = view, removed
        return self._code_views[node.id]

    def program_text(self, node, index, role, omit_idea=False):
        idea, idea_reason = self.idea_view(node)
        omissions = []
        if omit_idea and idea:
            idea, idea_reason = '', 'context_budget'
        if idea_reason:
            omissions.append({'node_id': node.id, 'kind': 'idea', 'reason': idea_reason})
        code, removed_comments = self.code_view(node)
        if removed_comments:
            omissions.append({
                'node_id': node.id, 'kind': 'code_comment',
                'reason': 'temporary_algorithm_reference', 'count': removed_comments,
            })
        return (f'Algorithm {index}\nRole: {role}\nFitness: {node.fitness}\nIdea: {idea}\n'
                f'Code:\n```python\n{code}\n```'), omissions

    def trajectory(self, parent, references, operator, donor=None, roles=None):
        nodes = sorted(([parent] if parent is not None else []) + list(references),
                       key=lambda node: (node.fitness, node.id))
        executed = 'Refine' if operator == 'Fuse' and donor is None else operator
        positions = {node.id: index for index, node in enumerate(nodes, 1)}
        roles = roles or {}
        instruction = TRAJECTORY_INSTRUCTIONS[executed].format(
            parent=positions.get(parent.id) if parent else None,
            donor=positions.get(donor.id) if donor else None,
        )
        def render(omit_idea=False):
            rendered = [self.program_text(
                node, index, roles.get(node.id, 'evidence_reference'), omit_idea,
            ) for index, node in enumerate(nodes, 1)]
            return [item[0] for item in rendered], [entry for item in rendered for entry in item[1]]
        blocks, omissions = render()
        def assemble():
            parts = [self.task_contract]
            if blocks:
                parts.append('# Design Evidence\n\n' + '\n\n'.join(blocks))
            parts.extend(['# Algorithm Design Task\n' + instruction, '# Output\n' + TRAJECTORY_OUTPUT])
            return '\n\n\n'.join(parts)
        text = assemble()
        if self.count(text, chat=True) > self.max_tokens:
            # Auxiliary design prose must not exclude otherwise fitting code.
            blocks, omissions = render(omit_idea=True)
            text = assemble()
        return text, nodes, donor, executed, blocks, omissions

    def fits_references(self, parent, references, operator, donor=None, roles=None):
        text, *_ = self.trajectory(parent, references, operator, donor, roles)
        return self.count(text, chat=True) <= self.max_tokens

    def fits(self, current, operator, donor=None):
        return self.fits_references(
            current, [donor] if donor else [], operator, donor,
        )

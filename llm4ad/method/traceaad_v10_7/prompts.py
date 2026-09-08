"""V10.7 task-specific evidence prompts with one Idea-and-Code output."""

import hashlib
import io
import re
import tokenize

from llm4ad.method.traceaad_v10_5.prompts import PromptBuilder as BaseBuilder
from llm4ad.method.traceaad_v10_6.prompts import INSTRUCTIONS, build_task_contract

GENERATION = 'idea_code_single_call_self_contained_v1'
TEMPORARY_REFERENCE_RE = re.compile(
    r'(?i)\b(?:Algorithm|Alg\.?)\s*#?\s*\d+\b|算法\s*#?\s*\d+'
)
#: Strict output contract. The parser only accepts a response that starts with
#: "Idea:" followed by a single python block, so the prompt must not invite
#: any preamble, analysis paragraph, or additional code block.
OUTPUT = (
    'Return exactly the following two parts and nothing else:\n\n'
    'Idea: <at most 100 words; a self-contained description of the main '
    'decision rule and key computation>\n'
    '```python\n<complete implementation>\n```\n\n'
    'Do not add headings, analysis, explanations, or additional code blocks. '
    'The Idea must stand on its own: state this algorithm\'s main decision rule and key '
    'computation without referring to Algorithm numbers or temporary display positions.'
)


TRAJECTORY_INSTRUCTIONS = {
    'Init': INSTRUCTIONS['Init'],
    'Refine': 'Improve the Design Base by testing one main improvement hypothesis. '
              'Use the observed transition as evidence when deciding what to change. '
              'Change the smallest coherent set of computations needed for that hypothesis. '
              'Leave unrelated parts unchanged unless the hypothesis requires otherwise.',
    'RefineBare': 'Improve the Design Base by testing one main improvement hypothesis. '
                  'Change the smallest coherent set of computations needed for that hypothesis. '
                  'Leave unrelated parts unchanged unless the hypothesis requires otherwise.',
    'Pivot': 'Design a competitive alternative main decision method for the task. '
             'Use the Comparison Baseline only as a comparison point, not as a template '
             'to inherit. Reuse shared utilities when useful, but the main decision '
             'criterion must be substantively different, not a cosmetic formula rewrite.',
    'Fuse': 'Design one coherent algorithm that uses useful computation from the Transfer '
            'Source to improve the Design Base. Aim to outperform the better input. '
            'Do not combine components merely because both are present.',
}
TRAJECTORY_OUTPUT = OUTPUT + '\nKeep the Idea within 100 words.'
IDEA_TOKENS = 256
#: Shown once above the evidence blocks: archived prose is unverified, code decides.
DESIGN_NOTE_LINE = (
    'Code is the authoritative implementation; the design note is an unverified description.'
)
#: Shown only with a real direct generation edge. Actionable, not epistemology.
TRANSITION_NOTE = (
    'Treat the fitness change as an outcome of the whole transition; '
    'inspect the code before reusing any changed component.'
)
TRAJECTORY_TEMPLATE_HASH = hashlib.sha256(
    (str(TRAJECTORY_INSTRUCTIONS) + TRAJECTORY_OUTPUT + DESIGN_NOTE_LINE
     + TRANSITION_NOTE + str(IDEA_TOKENS)).encode()
).hexdigest()

#: Prompt section titles. Roles are the only program identity; there are no
#: Algorithm numbers and no fitness ordering, so nothing can be mis-cited.
ROLE_TITLES = {
    'design_base': 'Design Base',
    'comparison_baseline': 'Comparison Baseline',
    'formation_evidence': 'Formation Evidence',
    'development_evidence': 'Development Evidence',
    'alternative_reference': 'Alternative Reference',
    'transfer_source': 'Transfer Source',
    'evidence_reference': 'Evidence',
}


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

    def code_view(self, node, strip_comments=False):
        if not hasattr(self, '_code_views'):
            self._code_views = {}
        key = (node.id, strip_comments)
        if key not in self._code_views:
            removed = 0
            try:
                tokens = []
                for token in tokenize.generate_tokens(io.StringIO(node.code).readline):
                    if token.type == tokenize.COMMENT and (
                            strip_comments or TEMPORARY_REFERENCE_RE.search(token.string)):
                        token = tokenize.TokenInfo(token.type, '', token.start, token.end, token.line)
                        removed += 1
                    tokens.append(token)
                view = tokenize.untokenize(tokens)
            except (IndentationError, tokenize.TokenError):
                view, removed = node.code, 0
            self._code_views[key] = view, removed
        return self._code_views[key]

    def program_text(self, node, role, omit_idea=False, strip_comments=False):
        title = ROLE_TITLES.get(role, role or 'Evidence')
        idea, idea_reason = self.idea_view(node)
        omissions = []
        if omit_idea and idea:
            idea, idea_reason = '', 'context_budget'
        if idea_reason:
            omissions.append({'node_id': node.id, 'kind': 'idea', 'reason': idea_reason})
        code, removed_comments = self.code_view(node, strip_comments)
        if removed_comments:
            omissions.append({
                'node_id': node.id, 'kind': 'code_comment',
                'reason': ('reference_comment_strip' if strip_comments
                           else 'temporary_algorithm_reference'),
                'count': removed_comments,
            })
        return (f'# {title}\nFitness: {node.fitness}\nDesign note: {idea}\n'
                f'Code:\n```python\n{code}\n```'), omissions

    @staticmethod
    def experiment_text(relations, roles):
        """Render only real direct generation edges.

        Archive references carry no relation section: their role title already
        says what they are, and genealogy disclaimers only distract from the
        design task. A Fuse edge whose donor is not shown says so without
        quoting the unseen donor's fitness, which is not actionable.
        """
        lines = []
        for relation in relations or []:
            if not relation.get('direct_generation_relation'):
                continue
            evidence = ROLE_TITLES.get(relation['kind'], relation['kind'])
            if relation['kind'] == 'formation_evidence':
                lines.append(
                    f"- {evidence} --{relation['operator']}--> Design Base. "
                    f"Fitness changed from {relation['source_fitness']} to "
                    f"{relation['target_fitness']} (delta {relation['fitness_delta']:+g})."
                )
            else:
                lines.append(
                    f"- Design Base --{relation['operator']}--> {evidence}. "
                    f"Fitness changed from {relation['source_fitness']} to "
                    f"{relation['target_fitness']} (delta {relation['fitness_delta']:+g})."
                )
            donor_id = relation.get('historical_donor_id')
            if donor_id is not None:
                donor_role = (roles or {}).get(donor_id)
                if donor_role is not None:
                    lines.append(
                        f'  A historical donor, {ROLE_TITLES.get(donor_role, donor_role)}, '
                        'also participated.'
                    )
                else:
                    lines.append(
                        '  This was a multi-input Fuse transition; another program also '
                        'contributed and is not shown.'
                    )
        if not lines:
            return ''
        return '# Observed Transition\n\n' + '\n'.join(lines) + '\n\n' + TRANSITION_NOTE

    def trajectory(self, parent, references, operator, donor=None, roles=None,
                   relations=None):
        # Fixed role order: the base first, then evidence in selection order.
        # No fitness sorting, so equal evidence renders byte-identical prompts.
        nodes = ([parent] if parent is not None else []) + list(references)
        executed = 'Refine' if operator == 'Fuse' and donor is None else operator
        roles = roles or {}
        if executed == 'Refine' and not references:
            instruction = TRAJECTORY_INSTRUCTIONS['RefineBare']
        else:
            instruction = TRAJECTORY_INSTRUCTIONS[executed]
        experiment = self.experiment_text(relations, roles)
        def render(omit_idea=False):
            rendered = [self.program_text(
                node, roles.get(node.id, 'evidence_reference'), omit_idea,
                strip_comments=node is not parent,
            ) for node in nodes]
            return [item[0] for item in rendered], [entry for item in rendered for entry in item[1]]
        blocks, omissions = render()
        def assemble():
            parts = [self.task_contract]
            if blocks:
                parts.append('# Design Evidence\n\n' + DESIGN_NOTE_LINE + '\n\n'
                             + '\n\n'.join(blocks))
            if experiment:
                parts.append(experiment)
            parts.extend(['# Design Task\n' + instruction, '# Output\n' + TRAJECTORY_OUTPUT])
            return '\n\n\n'.join(parts)
        text = assemble()
        if self.count(text, chat=True) > self.max_tokens:
            # Auxiliary design prose must not exclude otherwise fitting code.
            blocks, omissions = render(omit_idea=True)
            text = assemble()
        return text, nodes, donor, executed, blocks, omissions

    def fits_references(self, parent, references, operator, donor=None, roles=None,
                        relations=None):
        text, *_ = self.trajectory(
            parent, references, operator, donor, roles, relations,
        )
        return self.count(text, chat=True) <= self.max_tokens

    def fits(self, current, operator, donor=None):
        return self.fits_references(
            current, [donor] if donor else [], operator, donor,
        )

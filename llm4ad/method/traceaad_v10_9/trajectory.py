"""Complete direct trial evidence, conditioned on the requested design task."""

from difflib import unified_diff
from llm4ad.method.traceaad_v10_8.trajectory import TrajectoryBuilder as BaseBuilder, digest
from llm4ad.method.traceaad_v10_7.prompts import OUTPUT, GENERATION

CONTEXT_POLICY = 'task_trials_formation_and_aware_init_v2'
INITIALIZATION_POLICY = 'complete_prior_roots_temporal_spread_v1'
INSTRUCTIONS = {
    'Init': 'Design a competitive coherent decision method with meaningful task-dependent computations. '
            'When previous initial algorithms are shown, inspect their actual decision rules and propose '
            'another promising decision hypothesis. Make the difference affect actual choices or search '
            'behavior, rather than names, unused terms or score transformations that preserve decisions. '
            'You may reuse and reorganize useful components. References are comparison evidence, not a '
            'host that must be refined or a blacklist of computations. Do not sacrifice competitiveness '
            'merely to look different. Without references, design an independent initial algorithm.',
    'Refine': 'Develop the current main decision method. Inspect direct trials to identify a remaining '
              'weakness or an incompletely developed useful component. Test one concrete improvement '
              'hypothesis, preserving unrelated working computations. Numeric tuning is also valid. '
              'Do not repeat an observed trial without a materially different hypothesis.',
    'Tune': 'Keep the current algorithm structure, variable names, formulas, control flow and interfaces. '
            'Change only numeric constants to refine a small coherent set of influential coefficients, '
            'exponents, thresholds or schedules. Use measured direct trials to choose a new parameter '
            'setting; avoid repeating tested settings. Do not change index constants or safety limits '
            'unless they are deliberate algorithm parameters. Return the complete program.',
    'Pivot': 'Propose a different competitive main decision method or reorganize useful components '
             'into a new structure with room for further refinement. Inspect formation and direct '
             'trials to avoid merely renaming or reweighting the same rule. Reuse useful parts. '
             'A fitness plateau alone does not prove that a decision method is exhausted.',
    'Fuse': 'Keep the current implementation as the host algorithm. Identify a specific weakness '
            'and borrow a donor computation that addresses it; adapt its scale, inputs and interaction '
            'with the host. Preserve unrelated host logic and produce one coherent program, not an '
            'ensemble wrapper or a wholesale donor replacement. A weaker donor can contain a valuable '
            'component. If no donor component is useful, improve the host without forced mixing. '
            'Aim to outperform both inputs. Inspect actual code rather than copying a verbal idea.',
}
NOTE = ('# Observed Experiments\nEach block is an actual direct source-to-target trial, not a '
        'chronological chain between blocks. Fitness measures the whole program (higher is better); '
        'it does not identify the causal contribution of a component. No reported trial means no '
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
            parts.append(self.program(parent, 'Current Implementation'))
        if donor is not None:
            parts.append(self.program(donor, 'Donor'))
        if blocks:
            parts.append(NOTE + '\n\n' + '\n\n'.join(blocks))
        parts.extend(['# Design Task\n' + INSTRUCTIONS[operator], '# Output\n' + OUTPUT])
        return '\n\n\n'.join(parts)

    def program(self, node, title):
        return super().program(node, f'{title} (node {node.id})')

    def trial(self, source, target, role):
        # A predecessor alone cannot reconstruct an outgoing trial. Always show
        # its complete patch or complete target; both endpoints are then known.
        if target.parent_id != source.id:
            raise ValueError('trial must be a real direct edge')
        before, after = self.code_view(source)[0], self.code_view(target)[0]
        patch = ''.join(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n'
                        for line in unified_diff(before.splitlines(True), after.splitlines(True),
                                                 fromfile='source.py', tofile='target.py'))
        diff = f'Complete source-to-target diff:\n```diff\n{patch}```'
        # Both codes for disconnected formation edges; for direct outgoing
        # trials the source is the fully displayed current/donor program.
        text = (f'## {role}: node {source.id} -> node {target.id}\n'
                f'Evaluation: {source.evaluation_id} -> {target.evaluation_id}\n'
                f'Executed operator: {target.operator}; Fitness: {source.fitness} -> {target.fitness}\n')
        if target.donor_id is not None:
            text += f'Historical donor node {target.donor_id} also participated; this block does not isolate its effect.\n'
        if role != 'Direct trial':
            text += f'Complete source code:\n```python\n{before}\n```\n'
        text += diff
        full = (f'Complete target code:\n```python\n{after}\n```')
        if self.count(full) < self.count(diff):
            text = text[:-len(diff)] + full
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
        if parent is None:
            return []
        children = [n for n in self.all_nodes() if n.parent_id == parent.id]
        # Compare a best improving direct trial and latest non-improving trial.
        # Tune prioritizes previous Tune trials, but can use other direct trials.
        positive = sorted((n for n in children if n.fitness > parent.fitness),
                          key=lambda n: (operator == 'Tune' and n.operator == 'Tune', n.fitness, n.id), reverse=True)
        negative = sorted((n for n in children if n.fitness <= parent.fitness),
                          key=lambda n: (operator == 'Tune' and n.operator == 'Tune', n.id), reverse=True)
        edges = [(parent, group[0], 'Direct trial') for group in (negative, positive) if group]
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
        roots = sorted((n for n in self.all_nodes() if n.parent_id is None), key=lambda n: n.id)
        # Alternate newest/oldest to cover the observed collection when only a
        # subset fits. This is temporal coverage, not semantic diversity scoring.
        order = []
        left, right = 0, len(roots) - 1
        while left <= right:
            order.append(roots[right])
            right -= 1
            if left <= right:
                order.append(roots[left])
                left += 1
        chosen, omissions = [], []
        def render(nodes):
            parts = [self.task_contract]
            if nodes:
                parts.append('# Previous Initial Algorithms\n'
                             'These are complete previously evaluated programs. Their scores describe '
                             'whole-program performance, not the value of every component.')
                parts.extend(self.program(n, 'Previous initial algorithm') for n in sorted(nodes, key=lambda n: n.id))
            parts.extend(['# Design Task\n' + INSTRUCTIONS['Init'], '# Output\n' + OUTPUT])
            return '\n\n\n'.join(parts)
        text = render(chosen)
        if self.count(text, chat=True) > self.max_tokens:
            raise ValueError('minimum complete prompt exceeds the model context budget')
        for node in order:
            candidate = render([*chosen, node])
            if self.count(candidate, chat=True) <= self.max_tokens:
                chosen.append(node)
                text = candidate
            else:
                omissions.append({'node_id': node.id, 'reason': 'initial_reference_token_budget'})
        chosen.sort(key=lambda n: n.id)
        return text, {
            'initialization_policy': INITIALIZATION_POLICY,
            'initial_root_ids': [n.id for n in roots],
            'initial_reference_ids': [n.id for n in chosen],
            'history_ids': [], 'history_edge_count': 0, 'history_tokens': 0,
            'evidence_relations': [], 'context_omissions': omissions,
            'context_node_ids': [n.id for n in chosen],
            'context_code_views': [self.view_record(n) for n in chosen],
            'context_best_fitness': max((n.fitness for n in chosen), default=None),
        }

    def build(self, parent, operator, donor=None):
        if operator == 'Init':
            return self.build_initial()
        text = self.assemble(parent, operator, donor)
        if self.count(text, chat=True) > self.max_tokens:
            raise ValueError('minimum complete prompt exceeds the model context budget')
        blocks, relations, shown, omissions = [], [], {}, []
        for node in (parent, donor):
            if node is not None:
                shown[node.id] = node
        for source, target, role in self.evidence(parent, operator, donor):
            block, facts = self.trial(source, target, role)
            proposed = [*blocks, block]
            candidate = self.assemble(parent, operator, donor, proposed)
            if (self.count(NOTE + '\n\n' + '\n\n'.join(proposed)) > self.history_tokens or
                    self.count(candidate, chat=True) > self.max_tokens):
                omissions.append({'source_id': source.id, 'target_id': target.id, 'reason': 'token_budget'})
                continue
            blocks, text = proposed, candidate
            relations.append(facts)
            shown.update({source.id: source, target.id: target})
            if target.donor_id is not None:
                shown[target.donor_id] = self.lookup(target.donor_id)
        return text, {
            'history_ids': [r['target_id'] for r in relations], 'history_edge_count': len(relations),
            'history_tokens': self.count(NOTE + '\n\n' + '\n\n'.join(blocks)) if blocks else 0,
            'evidence_relations': relations, 'context_omissions': omissions,
            'context_node_ids': list(shown),
            'context_code_views': [self.view_record(n) for n in shown.values()],
            'context_best_fitness': max((n.fitness for n in shown.values()), default=None),
        }

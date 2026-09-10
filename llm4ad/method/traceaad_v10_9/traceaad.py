"""V10.9: bounded development, targeted transfer and dedicated numerical refinement."""

import ast
from functools import lru_cache
from pathlib import Path

from llm4ad.method.traceaad_v10_8.traceaad import TraceAADV108
from llm4ad.method.traceaad_v10_7.sampling import MAX_FIT_ATTEMPTS, _weighted_order
from llm4ad.method.traceaad_v10_5.traceaad import ess
from . import trajectory
from .trajectory import digest

OPERATOR_PROBABILITIES = {'Refine': 0.25, 'Tune': 0.25, 'Pivot': 0.15, 'Fuse': 0.35}
SELECTION_POLICY = 'ess8_recent_structure_development_v1'
DEDUP_POLICY = 'input_and_initial_root_ast_preserve_docstrings_v2'


@lru_cache(maxsize=8192)
def code_key(code, numeric=False):
    """Syntax signature, not a claim about semantics or idea identity."""
    tree = ast.parse(code)
    if numeric:
        class Numbers(ast.NodeTransformer):
            def visit_Constant(self, node):
                if type(node.value) in (int, float, complex):
                    return ast.copy_location(ast.Constant(value=0), node)
                return node

            def visit_UnaryOp(self, node):
                if (isinstance(node.op, (ast.USub, ast.UAdd)) and
                        isinstance(node.operand, ast.Constant) and
                        type(node.operand.value) in (int, float, complex)):
                    return ast.copy_location(ast.Constant(value=0), node)
                return self.generic_visit(node)
        tree = Numbers().visit(tree)
    return ast.dump(tree, include_attributes=False)


class TraceAADV109(TraceAADV108):
    METHOD = 'v109'
    DISPLAY_NAME = 'V10.9'
    STATE_VERSION = 1091
    OPERATOR_PROBABILITIES = OPERATOR_PROBABILITIES
    TEMPLATE_HASH = trajectory.TEMPLATE_HASH
    CONTEXT_POLICY = trajectory.CONTEXT_POLICY

    def __init__(self, **kwargs):
        # The formal policy is fixed; inherit C's attainable ESS handling.
        super().__init__(allocation_arm='C', **kwargs)
        if kwargs.get('task_name') == 'op_aco':
            self.task_contract += ('\n\nEvaluator execution facts: node 0 is masked as a candidate throughout '
                                   'ACO sampling. Return-to-depot distance enters feasibility checks, '
                                   'but changing only heuristic[:, 0] cannot change sampled moves. '
                                   'The heuristic is a static edge prior, computed before ACO runs.')
            self.mechanism['task_contract_hash'] = digest(self.task_contract)
        self.builder = trajectory.TrajectoryBuilder(
            self.llm, self.task_contract, max_tokens=self.builder.max_tokens,
            history_tokens=self.mechanism['history_tokens'], max_events=self.traj_gens,
            lookup=self.tree.nodes.get, all_nodes=self.tree.all_nodes,
            log_count=lambda r: self._append_record(self.run_dir / 'tokenizer_calls.jsonl', r),
        )
        self.mechanism.update(
            initialization_policy=trajectory.INITIALIZATION_POLICY,
            operator_probabilities=self.OPERATOR_PROBABILITIES,
            context_policy=self.CONTEXT_POLICY, selection_policy=SELECTION_POLICY,
            dedup_policy=DEDUP_POLICY, allocation_policy=SELECTION_POLICY,
            development_probability=0.2, development_window=64,
            development_attempts=2, donor_uniform_probability=0.5,
            tune_policy='prompt_numeric_only_observed_ast_classification_v1',
        )
        for source in (Path(__file__), Path(trajectory.__file__)):
            self.mechanism['source_hashes'][str(source.resolve())] = digest(source.read_text())

    def eligible_nodes(self):
        for node in self.tree.all_nodes():
            if node.id not in self._eligible:
                self._eligible[node.id] = all(self.builder.fits(node, op)
                                              for op in ('Refine', 'Tune', 'Pivot'))
        nodes = [n for n in self.tree.all_nodes() if self._eligible[n.id]]
        if not nodes:
            raise ValueError('no archived parent fits the complete model context')
        return nodes

    def node_distribution(self, nodes, operator):
        probabilities, metadata = super().node_distribution(nodes, operator)
        recent = []
        if operator != 'Pivot':
            for i, node in enumerate(nodes):
                parent = self.tree.nodes.get(node.parent_id)
                if (parent is not None and
                        0 <= self.budget_used - node.evaluation_id < 64 and
                        self.parent_selection_counts.get(node.id, 0) < 2 and
                        code_key(node.code, True) != code_key(parent.code, True)):
                    recent.append(i)
            if recent:
                indices = set(recent)
                probabilities = [0.8 * p + (0.2 / len(recent) if i in indices else 0)
                                 for i, p in enumerate(probabilities)]
        metadata.update(development_node_ids=[nodes[i].id for i in recent],
                        development_probability=0.2 if recent else 0.0,
                        conditional_ess=ess(probabilities), allocation_policy=SELECTION_POLICY)
        return probabilities, metadata

    def select_donor(self, parent):
        nodes = [n for n in self.tree.all_nodes()
                 if n.id != parent.id and code_key(n.code) != code_key(parent.code)]
        if not nodes:
            return None, []
        # Include weak whole programs. Relevance is assessed in the transfer task,
        # not inferred from AST distance or the donor's total score.
        quality, _ = super().node_distribution(nodes, 'Fuse')
        weights = [0.5 * p + 0.5 / len(nodes) for p in quality]
        attempts = []
        for donor in _weighted_order(nodes, weights, self.rng, MAX_FIT_ATTEMPTS):
            fits = self.builder.fits(parent, 'Fuse', donor)
            attempts.append({'node_id': donor.id, 'fitness': donor.fitness, 'fits': fits})
            if fits:
                return donor, attempts
        return None, attempts

    def _duplicate_inputs(self, code):
        if self.pending['operator'] == 'Init':
            return [{'role': 'existing_root', 'node_id': n.id, 'view': 'ast'}
                    for n in self.tree.all_nodes() if n.parent_id is None and code_key(code) == code_key(n.code)]
        return [{'role': role, 'node_id': self.pending[role + '_id'], 'view': 'ast'}
                for role in ('parent', 'donor')
                if self.pending[role + '_id'] is not None and
                code_key(code) == code_key(self.tree.nodes[self.pending[role + '_id']].code)]

    def _append_record(self, path, record):
        if path == self.events_path and record.get('operator') == 'Init' and record.get('status') == 'duplicate_code':
            record['reason'] = 'identical_to_existing_root'
        if path == self.events_path and record.get('parent_id') is not None:
            completion = self.pending['completion']
            parsed = self.parse_response(completion['response'], completion['finish_reason'])
            if parsed is not None:
                code = parsed[1]
                parent = self.tree.nodes[record['parent_id']]
                record['edit_kind'] = ('same' if code_key(code) == code_key(parent.code) else
                                       'numeric_only' if code_key(code, True) ==
                                       code_key(parent.code, True) else 'structural')
        super()._append_record(path, record)

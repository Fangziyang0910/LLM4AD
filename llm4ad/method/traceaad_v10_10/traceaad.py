"""V10.10: quality-based search with formation history and one error-conditioned repair."""

import ast
import json
import math
import time
import traceback
from functools import lru_cache
from pathlib import Path

from llm4ad.method.traceaad_v10_8.traceaad import TraceAADV108
from llm4ad.method.traceaad_v10_5.traceaad import read_journal, UnknownEvaluation
from . import trajectory
from .trajectory import digest
from . import errors

OPERATOR_PROBABILITIES = {'Refine': 0.25, 'Tune': 0.25, 'Pivot': 0.25, 'Fuse': 0.25}
REPAIRABLE_FAILURES = {'exec_error', 'runtime_error', 'timeout',
                       'invalid_result', 'nonfinite_fitness'}
SELECTION_POLICY = 'ess8_quality_only_v1'
DEDUP_POLICY = 'parent_donor_ast_preserve_docstrings_v1'
ERROR_HANDLING = 'candidate_error_one_repair_v2'


@lru_cache(maxsize=8192)
def code_key(code):
    """Exact syntax signature, preserving docstrings and numeric constants."""
    return ast.dump(ast.parse(code), include_attributes=False)


class TraceAADV1010(TraceAADV108):
    METHOD = 'v1010'
    DISPLAY_NAME = 'V10.10'
    STATE_VERSION = 10100
    OPERATOR_PROBABILITIES = OPERATOR_PROBABILITIES
    TEMPLATE_HASH = trajectory.TEMPLATE_HASH
    CONTEXT_POLICY = trajectory.CONTEXT_POLICY

    def __init__(self, **kwargs):
        # The formal policy is fixed; inherit C's attainable ESS handling.
        super().__init__(allocation_arm='C', **kwargs)
        notes = getattr(self.evaluation, 'design_notes', '').strip()
        if notes:
            self.task_contract += f'\n\n# Evaluator Semantics\n{notes}'
        self.builder = trajectory.TrajectoryBuilder(
            self.llm, self.task_contract,
            max_tokens=self.max_context_tokens - self.mechanism['context_margin'] - 1,
            history_tokens=self.mechanism['history_tokens'], max_events=self.traj_gens,
            lookup=self.tree.nodes.get, all_nodes=self.tree.all_nodes,
            log_count=lambda r: self._append_record(self.run_dir / 'tokenizer_calls.jsonl', r),
        )
        self.mechanism.update(
            initialization_policy=trajectory.INITIALIZATION_POLICY,
            operator_probabilities=self.OPERATOR_PROBABILITIES,
            context_policy=self.CONTEXT_POLICY, selection_policy=SELECTION_POLICY,
            dedup_policy=DEDUP_POLICY, allocation_policy=SELECTION_POLICY,
            donor_uniform_probability=0.5,
            tune_policy='parameter_settings_v2',
            task_contract_hash=digest(self.task_contract),
        )
        for source in (Path(__file__), Path(trajectory.__file__)):
            self.mechanism['source_hashes'][str(source.resolve())] = digest(source.read_text())
        self.mechanism.update(generation=trajectory.GENERATION,
                              error_handling=ERROR_HANDLING, max_repairs=1)
        for source in (Path(errors.__file__),):
            self.mechanism['source_hashes'][str(source.resolve())] = digest(source.read_text())
        for key in ('ess_fraction', 'ess_minimum', 'history_tokens'):
            self.mechanism['inherited_unused'][key] = self.mechanism.pop(key)
        self.mechanism.pop('reference_fit_attempts', None)
        events = read_journal(self.events_path)
        self._last_event = events[-1] if events else None

    def _generate_pending(self):
        # Reserve only the output space still available for this complete input.
        configured = self.output_tokens
        available = (self.max_context_tokens - self.mechanism['context_margin']
                     - self.pending['prompt_tokens'])
        if available < 1:
            raise ValueError('complete input leaves no room for model output')
        self.output_tokens = min(configured, available)
        try:
            super()._generate_pending()
        finally:
            self.output_tokens = configured

    def parse_response(self, response, finish_reason='unknown'):
        parsed, mode, error = errors.parse_candidate(response, finish_reason, super().parse_response)
        self._parse_diagnostics = (mode, error)
        return parsed

    def _schedule(self):
        previous = self._last_event
        if (previous and previous['candidate_id'] == self.completed_attempts and
                previous['status'] == 'eval_failed' and
                previous['reason'] not in REPAIRABLE_FAILURES):
            raise RuntimeError(
                f"evaluation infrastructure failed: {previous['reason']}: "
                f"{previous.get('error') or 'unknown error'}"
            )
        if (previous and previous['candidate_id'] == self.completed_attempts and
                (previous['status'] == 'invalid_output' or
                 previous['reason'] in REPAIRABLE_FAILURES) and
                not previous.get('repair_of')):
            # Completed failures are durable events. A repair gets its own candidate and
            # receipt, so the existing crash recovery and actual-call budget apply unchanged.
            with self.llm_calls_path.open() as handle:
                responses = (json.loads(line) for line in handle)
                response = next(r['response'] for r in responses
                                if r['candidate_id'] == previous['candidate_id'] and 'response' in r)
            text = errors.repair_prompt(self.task_contract, response, previous)
            tokens = self.builder.count(text, chat=True)
            return {
                'candidate_id': self.completed_attempts + 1, 'phase': 'selected',
                'repair_of': previous['candidate_id'], 'generation_kind': 'repair',
                **{k: previous[k] for k in ('operator', 'requested_operator', 'parent_id',
                    'donor_id', 'parent_fitness', 'donor_fitness', 'selection', 'operator_probabilities')},
                'best_before': self.tree.best().fitness if self.tree.nodes else None,
                'prompt': text, 'prompt_tokens': tokens, 'prompt_hash': digest(text),
                'template_hash': self.TEMPLATE_HASH, 'context_policy': 'failed_output_and_error_v1',
                'context_best_fitness': previous['parent_fitness'],
                'rng_state': list(self.rng.getstate()), 'llm_attempts': 0,
            }
        return super()._schedule()

    def _log_call(self, record):
        record['stage'] = 'repair' if self.pending and self.pending.get('repair_of') else 'generation'
        if self.pending and self.pending.get('repair_of'):
            record['repair_of'] = self.pending['repair_of']
        super()._log_call(record)

    def _evaluate_pending(self, parsed):
        p = self.pending
        if p['phase'] == 'evaluating':
            outcome = self._outcomes.get(p['candidate_id'])
            if outcome is None:
                raise UnknownEvaluation(f"candidate {p['candidate_id']} has an unknown evaluation reservation; refusing to repeat it")
        else:
            p.update(phase='evaluating', evaluation_id=self.budget_used + 1)
            self._persist_pending()
            started = time.time()
            fitness, reason, error_type, error, trace = None, None, None, None, None
            try:
                result = self.secure.evaluate_program_with_details(parsed[2])
                reason, error_type, error, trace = result.failure_kind, result.error_type, result.error, result.traceback
                if result.result is not None:
                    value = float(result.result)
                    if math.isfinite(value):
                        fitness = value
                    else:
                        reason, error = 'nonfinite_fitness', 'Evaluator returned a nonfinite fitness.'
            except Exception as exc:
                reason, error_type, error, trace = 'evaluation_error', type(exc).__name__, str(exc), traceback.format_exc()
            outcome = dict(candidate_id=p['candidate_id'], evaluation_id=p['evaluation_id'],
                           fitness=fitness, reason=reason, error_type=error_type, error=error,
                           traceback=trace, eval_seconds=time.time() - started,
                           repair_of=p.get('repair_of'))
            self._append_record(self.evaluations_path, outcome)
            self._outcomes[p['candidate_id']] = outcome
        p.update(phase='evaluated', outcome=outcome)
        self._persist_pending()

    def eligible_nodes(self):
        return self.tree.all_nodes()

    def select_donor(self, parent):
        nodes = [n for n in self.tree.all_nodes()
                 if n.id != parent.id and code_key(n.code) != code_key(parent.code)]
        if not nodes:
            return None, []
        # Include weak whole programs. Relevance is assessed in the transfer task,
        # not inferred from AST distance or the donor's total score.
        quality, _ = super().node_distribution(nodes, 'Fuse')
        weights = [0.5 * p + 0.5 / len(nodes) for p in quality]
        donor = self.rng.choices(nodes, weights=weights)[0]
        return donor, [{'node_id': donor.id, 'fitness': donor.fitness}]

    def _duplicate_inputs(self, code):
        # Search-stage input-copy filter; initialization accepts any valid program.
        return [{'role': role, 'node_id': self.pending[role + '_id'], 'view': 'ast'}
                for role in ('parent', 'donor')
                if self.pending[role + '_id'] is not None and
                code_key(code) == code_key(self.tree.nodes[self.pending[role + '_id']].code)]

    def _append_record(self, path, record):
        if path == self.events_path:
            record['parse_mode'], parse_error = self._parse_diagnostics
            if record['status'] == 'invalid_output':
                record.update(error_type='ParseError', error=parse_error)
            elif record['status'] == 'eval_failed':
                record.update({k: self.pending['outcome'].get(k) for k in ('error_type', 'error', 'traceback')})
        super()._append_record(path, record)
        if path == self.events_path:
            self._last_event = record

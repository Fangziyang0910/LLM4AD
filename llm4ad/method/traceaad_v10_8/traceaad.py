"""V10.8: grouped opportunities and verified formation-conditioned generation."""

import json
import math
from pathlib import Path
import time
import traceback

from llm4ad.method.traceaad_v10_3.schema import SearchTree
from llm4ad.method.traceaad_v10_3.traceaad import TraceAADV103, calibrate_beta
from llm4ad.method.traceaad_v10_5.traceaad import atomic_json, ess, UnknownEvaluation
from llm4ad.method.traceaad_v10_7 import traceaad as v107
from llm4ad.method.traceaad_v10_7.sampling import (
    MAX_FIT_ATTEMPTS, _task_base_weights, _weighted_order,
)
from . import trajectory
from .trajectory import digest

GROUP_POLICY = 'exact_code_first_fitness_v1'


def code_groups(nodes):
    groups = {}
    for node in sorted(nodes, key=lambda n: (n.evaluation_id, n.id)):
        if math.isfinite(node.fitness):
            groups.setdefault(node.code, []).append(node)
    return groups


class ImplementationTree(SearchTree):
    def best(self):
        # Keep every measured score untouched; select by first valid evaluation.
        return max((records[0] for records in code_groups(self.all_nodes()).values()),
                   key=lambda node: node.fitness)


class TraceAADV108(v107.TraceAADV107):
    METHOD = 'v108'

    def __init__(self, *, history_tokens=8192, **kwargs):
        if 'max_context_programs' in kwargs:
            raise TypeError('V10.8 uses a formation suffix and one optional Fuse donor')
        super().__init__(history_tokens=history_tokens, **kwargs)
        self.tree = ImplementationTree()
        self.builder = trajectory.TrajectoryBuilder(
            self.llm, self.task_contract, max_tokens=self.builder.max_tokens,
            history_tokens=history_tokens, max_events=self.traj_gens,
            lookup=self.tree.nodes.get,
            log_count=lambda record: self._append_record(
                self.run_dir / 'tokenizer_calls.jsonl', record),
        )
        self.mechanism['inherited_unused'] = {'donor_topk': self.donor_topk}
        self.mechanism.pop('max_context_programs')
        self.mechanism.update(
            history_tokens=history_tokens, traj_gens=self.traj_gens,
            context_policy=trajectory.CONTEXT_POLICY, group_policy=GROUP_POLICY,
            seed=kwargs.get('seed', 0),
        )
        # Public JSON configuration covers task sizes, panel seeds and evaluator
        # execution limits; large private dataset arrays stay out of checkpoints.
        config = {}
        for key, value in vars(self.evaluation).items():
            if key.startswith('_'):
                continue
            try:
                config[key] = json.loads(json.dumps(value, allow_nan=False))
            except (TypeError, ValueError):
                continue
        self.mechanism['evaluation_config'] = config
        for source in (Path(__file__), Path(trajectory.__file__), Path(v107.__file__)):
            self.mechanism['source_hashes'][str(source.resolve())] = digest(source.read_text())

    def eligible_groups(self):
        groups = code_groups(self.tree.all_nodes())
        eligible = {n.id for n in self.eligible_nodes()}
        # Quality always comes from the first record, even if only a later
        # record's measured-fitness text fits the input window.
        return [(records[0], [n for n in records if n.id in eligible],
                 sum(self.parent_selection_counts.get(n.id, 0) for n in records))
                for records in groups.values() if any(n.id in eligible for n in records)]

    def group_distribution(self, groups, operator):
        scores = [first.fitness for first, _, _ in groups]
        beta, attainable_target, quality_ess = calibrate_beta(
            scores, self.ess_fraction, self.ess_minimum)
        maximum = max(scores)
        weights = [math.exp(beta * (first.fitness - maximum)) / math.sqrt(1 + count)
                   for first, _, count in groups]
        total = sum(weights)
        p0 = [w / total for w in weights]
        probabilities = ([0.5 * p + 0.5 / len(groups) for p in p0]
                         if operator == 'Pivot' else p0)
        return probabilities, {
            'eligible_groups': len(groups), 'beta': beta,
            'ess_target': min(len(groups), max(self.ess_fraction * len(groups), self.ess_minimum)),
            'attainable_ess_target': attainable_target, 'quality_ess': quality_ess,
            'corrected_ess': ess(p0), 'conditional_ess': ess(probabilities),
        }

    def select_donor(self, parent):
        groups = code_groups(n for n in self.tree.all_nodes() if n.code != parent.code)
        firsts = [records[0] for records in groups.values()]
        weights = _task_base_weights(firsts, {}, 'Fuse')
        attempts = []
        for first in _weighted_order(firsts, weights, self.rng, MAX_FIT_ATTEMPTS):
            donor = self.rng.choice(groups[first.code])
            fits = self.builder.fits(parent, 'Fuse', donor)
            attempts.append({'node_id': donor.id, 'group_code_hash': digest(first.code),
                             'group_fitness': first.fitness, 'fits': fits})
            if fits:
                return donor, attempts
        return None, attempts

    def _schedule(self):
        started = time.monotonic()
        counts_before = len(self.builder._counts)
        parent = donor = None
        requested = operator = 'Init'
        selection, donor_attempts = {}, []
        if len(self.tree.roots) >= self.n_roots:
            requested = self.rng.choices(
                list(v107.OPERATOR_PROBABILITIES),
                weights=list(v107.OPERATOR_PROBABILITIES.values()))[0]
            operator = requested
            groups = self.eligible_groups()
            probabilities, selection = self.group_distribution(groups, requested)
            index = self.rng.choices(range(len(groups)), weights=probabilities)[0]
            first, records, group_count = groups[index]
            parent = self.rng.choice(records)
            selection.update(
                parent_route='operator_then_code_group', group_code_hash=digest(first.code),
                group_fitness=first.fitness, group_count_before=group_count,
                group_probability=probabilities[index], group_record_count=len(records),
                parent_probability=probabilities[index] / len(records),
                parent_count_before=self.parent_selection_counts.get(parent.id, 0),
            )
            if requested == 'Fuse':
                donor, donor_attempts = self.select_donor(parent)
                if donor is None:
                    operator = 'Refine'
                    selection['fallback_reason'] = (
                        'donor_fit_attempt_limit' if len(donor_attempts) == MAX_FIT_ATTEMPTS
                        else 'no_fitting_distinct_code_donor')
        text, context = self.builder.build(parent, operator, donor)
        tokens = self.builder.count(text, chat=True)
        # Count only a complete selected request. Tokenizer failures before this
        # point are not generation opportunities; pending recovery restores once.
        if parent is not None:
            self.parent_selection_counts[parent.id] = selection['parent_count_before'] + 1
        return {
            'candidate_id': self.completed_attempts + 1, 'phase': 'selected',
            'requested_operator': requested, 'operator': operator,
            'parent_id': parent.id if parent else None, 'donor_id': donor.id if donor else None,
            'parent_fitness': parent.fitness if parent else None,
            'donor_fitness': donor.fitness if donor else None,
            'best_before': self.tree.best().fitness if self.tree.nodes else None,
            'operator_probabilities': v107.OPERATOR_PROBABILITIES, 'selection': selection,
            'donor_attempts': donor_attempts, 'prompt': text, 'prompt_tokens': tokens,
            'prompt_hash': digest(text), 'template_hash': trajectory.TEMPLATE_HASH,
            'context_policy': trajectory.CONTEXT_POLICY, **context,
            'scheduling_seconds': time.monotonic() - started,
            'tokenizer_requests': len(self.builder._counts) - counts_before,
            'rng_state': list(self.rng.getstate()), 'llm_attempts': 0,
        }

    def _append_record(self, path, record):
        if path == self.events_path:
            best = self.tree.best().fitness if self.tree.nodes else None
            record['best_so_far'] = best
            node = self.tree.nodes.get(record.get('node_id'))
            if node is not None:
                first = code_groups(self.tree.all_nodes())[node.code][0]
                record['implementation_fitness'] = first.fitness
                if record.get('best_before') is not None:
                    delta = first.fitness - record['best_before']
                    record.update(implementation_frontier_delta=delta,
                                  implementation_frontier_improved=delta > 0)
        super()._append_record(path, record)

    def _save_state(self):
        atomic_json(self.state_path, {
            'version': 1080, 'mechanism': self.mechanism, 'started_at': self.started_at,
            'nodes': self.tree.to_state(), 'rng_state': list(self.rng.getstate()),
            'parent_selection_counts': self.parent_selection_counts,
            'step_counter': self.step_counter, 'batch_counter': self.step_counter,
            'budget_used': self.budget_used, 'completed_attempts': self.completed_attempts,
            'invalid_streak': self._invalid_streak,
        })

    def _load_state(self):
        state = json.loads(self.state_path.read_text())
        if state.get('version') != 1080 or state.get('mechanism') != self.mechanism:
            raise ValueError('checkpoint mechanism/source/backend differs from this V10.8 configuration')
        TraceAADV103._load_state(self)
        self.completed_attempts = state['completed_attempts']
        self._invalid_streak = state['invalid_streak']
        if self.pending_path.exists():
            pending = json.loads(self.pending_path.read_text())
            if pending['candidate_id'] <= self.completed_attempts:
                self.pending_path.unlink()
                return
            if pending['candidate_id'] != self.completed_attempts + 1:
                raise ValueError('pending candidate is not the next attempt')
            self.pending = pending
            rng = pending['rng_state']
            self.rng.setstate((rng[0], tuple(rng[1]), rng[2]))
            if pending['parent_id'] is not None:
                self.parent_selection_counts[pending['parent_id']] = (
                    pending['selection']['parent_count_before'] + 1)

    def _write_summary(self, status, error=None):
        super()._write_summary(status, error)
        payload = json.loads(self.summary_path.read_text())
        groups = code_groups(self.tree.all_nodes())
        payload.update(group_policy=GROUP_POLICY, num_code_groups=len(groups),
                       fitness_instability=[
                           {'code_hash': digest(code), 'first_fitness': records[0].fitness,
                            'measurements': [{'node_id': n.id, 'evaluation_id': n.evaluation_id,
                                              'fitness': n.fitness} for n in records]}
                           for code, records in groups.items()
                           if any(n.fitness != records[0].fitness for n in records)
                       ])
        atomic_json(self.summary_path, payload)

    def run(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Reject a foreign configuration before writing any run artifacts.
        if self.state_path.exists():
            self._load_state()
        else:
            self._save_state()
        try:
            while self.budget_used < self.budget:
                self._advance()
                best = self.tree.best().fitness if self.tree.nodes else None
                print(f'v108: budget={self.budget_used}/{self.budget} '
                      f'nodes={len(self.tree.nodes)} best={best}', flush=True)
            if len(self.tree.roots) < self.n_roots:
                raise RuntimeError(f'evaluation budget exhausted before {self.n_roots} requested valid roots')
            self._write_summary('finished')
        except UnknownEvaluation:
            self._write_summary('blocked', error=traceback.format_exc())
            raise
        except KeyboardInterrupt:
            self._write_summary('interrupted')
            raise
        except Exception:
            self._write_summary('error', error=traceback.format_exc())
            raise

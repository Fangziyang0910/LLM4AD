"""V10.7 removes V10.6's independent implementation-Idea call."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import traceback

from llm4ad.method.traceaad_v10_3.traceaad import TraceAADV103
from llm4ad.method.traceaad_v10_5.traceaad import (
    TraceAADV105, UnknownEvaluation, atomic_json, ess,
)
from llm4ad.method.traceaad_v10_6 import prompts as v106_prompts
from llm4ad.method.traceaad_v10_6 import traceaad as v106_traceaad
from llm4ad.method.traceaad_v10_6.traceaad import (
    OPERATOR_PROBABILITIES, TraceAADV106, joint_parent_distribution,
)
from . import prompts


class TraceAADV107(TraceAADV106):
    METHOD = 'v107'

    def __init__(self, *, history_tokens=8192, task_name=None, **kwargs):
        TraceAADV105.__init__(self, history_tokens=history_tokens, **kwargs)
        self.task_contract = prompts.build_task_contract(self.evaluation)
        self.builder = prompts.PromptBuilder(
            self.llm, self.task_contract, max_tokens=self.builder.max_tokens,
            history_tokens=history_tokens, max_events=self.traj_gens,
            lookup=self.tree.nodes.get,
            log_count=lambda record: self._append_record(
                self.run_dir / 'tokenizer_calls.jsonl', record
            ),
        )
        self.mechanism.update(
            task_name=task_name, generation=prompts.GENERATION,
            task_contract_hash=hashlib.sha256(self.task_contract.encode()).hexdigest(),
        )
        for source in [
            Path(__file__), Path(prompts.__file__),
            Path(v106_traceaad.__file__), Path(v106_prompts.__file__),
        ]:
            self.mechanism['source_hashes'][str(source.resolve())] = hashlib.sha256(
                source.read_bytes()
            ).hexdigest()

    def _log_call(self, record):
        TraceAADV105._log_call(self, record)

    def _schedule(self):
        parent = donor = None
        selection = {}
        requested = operator = 'Init'
        if len(self.tree.roots) >= self.n_roots:
            nodes = self.eligible_nodes()
            p0, _, selection = self.parent_distribution(nodes, 'Refine')
            marginal, conditional = joint_parent_distribution(p0)
            index = self.rng.choices(range(len(nodes)), weights=marginal)[0]
            requested = self.rng.choices(
                list(conditional[index]), weights=list(conditional[index].values())
            )[0]
            operator = requested
            selection.update(
                parent_marginal_ess=ess(marginal),
                operator_conditional=conditional[index],
            )
            parent = nodes[index]
            count = self.parent_selection_counts.get(parent.id, 0)
            self.parent_selection_counts[parent.id] = count + 1
            selection.update(
                parent_route='joint_marginal', parent_probability=marginal[index],
                parent_count_before=count,
            )
            if requested == 'Fuse':
                donors = self.fitting_donors(parent)
                if donors:
                    donor = self.rng.choice(donors)
                else:
                    operator = 'Refine'
                    selection['fallback_reason'] = 'no fitting cross-lineage donor'
        ancestors = self.tree.ancestors(parent.id) if parent else []
        prompt = self.builder.build(parent, ancestors, operator, donor)
        return {
            'candidate_id': self.completed_attempts + 1, 'phase': 'selected',
            'requested_operator': requested, 'operator': operator,
            'parent_id': parent.id if parent else None,
            'donor_id': donor.id if donor else None,
            'operator_probabilities': OPERATOR_PROBABILITIES, 'selection': selection,
            'parent_fitness': parent.fitness if parent else None,
            'donor_fitness': donor.fitness if donor else None,
            'best_before': self.tree.best().fitness if self.tree.nodes else None,
            'prompt': prompt.text, 'prompt_tokens': prompt.tokens,
            'prompt_hash': hashlib.sha256(prompt.text.encode()).hexdigest(),
            'template_hash': prompts.TEMPLATE_HASH,
            'history_ids': prompt.history_ids,
            'context_omissions': prompt.omissions,
            'history_tokens': prompt.history_tokens,
            'rng_state': list(self.rng.getstate()), 'llm_attempts': 0,
        }

    def _advance(self) -> None:
        if self.pending is None:
            self.pending = self._schedule()
            self._persist_pending()
        pending = self.pending
        if pending['phase'] == 'selected':
            self._generate_pending()
        self._log_call(pending['completion'])
        response = pending['completion']
        parsed = self.parse_response(response['response'], response['finish_reason'])
        node = None
        reason = None
        if parsed is None:
            reason = ('length_truncated' if response['finish_reason'] == 'length'
                      else 'invalid_code_or_signature')
            self._invalid_streak += 1
            status = 'invalid_output'
        else:
            self._invalid_streak = 0
            if pending['phase'] != 'evaluated':
                self._evaluate_pending(parsed)
            outcome = pending['outcome']
            if outcome['evaluation_id'] != self.budget_used + 1:
                raise ValueError('evaluation receipt is not the next budget slot')
            self.budget_used = outcome['evaluation_id']
            reason = outcome['reason']
            status = 'eval_failed' if outcome['fitness'] is None else 'ok'
            if outcome['fitness'] is not None:
                node = self.tree.add(
                    code=parsed[1], idea=parsed[0], fitness=outcome['fitness'],
                    evaluation_id=self.budget_used, parent_id=pending['parent_id'],
                    operator=pending['operator'], donor_id=pending['donor_id'],
                )
        if pending['parent_id'] is not None:
            self.step_counter += 1
        record = {
            key: value for key, value in pending.items()
            if key not in ['prompt', 'rng_state', 'completion', 'phase', 'outcome']
        }
        record.update(
            ts=datetime.now().isoformat(timespec='seconds'), step=self.step_counter,
            status=status, reason=reason, budget_used=self.budget_used,
            evaluation_id=pending.get('outcome', {}).get('evaluation_id'),
            eval_seconds=pending.get('outcome', {}).get('eval_seconds'),
            llm_seconds=response['seconds'], node_id=node.id if node else None,
            fitness=node.fitness if node else None,
        )
        if node is not None and pending['parent_id'] is not None:
            record.update(
                parent_improved=node.fitness > pending['parent_fitness'],
                frontier_improved=node.fitness > pending['best_before'],
                parent_delta=node.fitness - pending['parent_fitness'],
                frontier_delta=node.fitness - pending['best_before'],
            )
            if pending['donor_id'] is not None:
                baseline = max(pending['parent_fitness'], pending['donor_fitness'])
                record['both_improved'] = node.fitness > baseline
                record['both_delta'] = node.fitness - baseline
        if pending['candidate_id'] not in self._logged_events:
            self._append_record(self.events_path, record)
            self._logged_events.add(pending['candidate_id'])
        self.completed_attempts = pending['candidate_id']
        self._save_state()
        self.pending_path.unlink(missing_ok=True)
        self.pending = None
        if self._invalid_streak >= 50:
            raise RuntimeError('50 consecutive generations produced no valid output')

    def _save_state(self) -> None:
        atomic_json(self.state_path, {
            'version': 107, 'mechanism': self.mechanism, 'started_at': self.started_at,
            'nodes': self.tree.to_state(), 'rng_state': list(self.rng.getstate()),
            'parent_selection_counts': self.parent_selection_counts,
            'step_counter': self.step_counter, 'batch_counter': self.step_counter,
            'budget_used': self.budget_used, 'completed_attempts': self.completed_attempts,
            'invalid_streak': self._invalid_streak,
        })

    def _load_state(self) -> None:
        state = json.loads(self.state_path.read_text())
        if state.get('version') != 107 or state.get('mechanism') != self.mechanism:
            raise ValueError('checkpoint mechanism/source/backend differs from this V10.7 configuration')
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
                    pending['selection']['parent_count_before'] + 1
                )

    def run(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        try:
            if self.state_path.exists():
                self._load_state()
            else:
                self._save_state()
            while self.budget_used < self.budget:
                self._advance()
                best = self.tree.best().fitness if self.tree.nodes else None
                print(
                    f'v107: budget={self.budget_used}/{self.budget} '
                    f'nodes={len(self.tree.nodes)} best={best}',
                    flush=True,
                )
            if len(self.tree.roots) < self.n_roots:
                raise RuntimeError(
                    f'evaluation budget exhausted before {self.n_roots} requested valid roots'
                )
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

"""TraceAAD V10.11: compact function-level search engine."""

import ast
import copy
import json
import math
import os
import random
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from functools import lru_cache
from pathlib import Path

from llm4ad.base import SecureEvaluator, TextFunctionProgramConverter
from . import errors, trajectory
from .core import (SearchTree, UnknownEvaluation, atomic_json, calibrate_beta, digest,
                   Node, ess, read_journal)

OPERATORS = ("Refine", "Tune", "Pivot", "Fuse")
OPERATOR_PROBABILITIES = {operator: 0.25 for operator in OPERATORS}
REPAIRABLE_FAILURES = {"exec_error", "runtime_error", "timeout", "invalid_result", "nonfinite_fitness"}


@lru_cache(maxsize=8192)
def code_key(code):
    return ast.dump(ast.parse(code), include_attributes=False)


class TraceAADV1011:
    METHOD = "v1011"

    def __init__(self, *, evaluation, llm, run_dir, budget=1000, n_roots=8,
                 traj_gens=8, output_tokens=16384, max_context_tokens=32768,
                 context_margin=256, history_code=False, seed=0):
        if budget < n_roots or n_roots < 1 or traj_gens < 0:
            raise ValueError("invalid budget, root, or history settings")
        self.evaluation, self.llm = evaluation, llm
        self.run_dir = Path(run_dir)
        self.budget, self.n_roots, self.traj_gens = budget, n_roots, traj_gens
        self.output_tokens, self.max_context_tokens = output_tokens, max_context_tokens
        self.context_margin = context_margin
        self.history_code = history_code
        self.secure = SecureEvaluator(evaluation)
        self._template_program = evaluation.template_program
        template = TextFunctionProgramConverter.text_to_program(self._template_program)
        if template is None or len(template.functions) != 1:
            raise ValueError("evaluation template must define exactly one function")
        self._template_func = template.functions[0]
        self._parse_interface = errors.expected_interface(self._template_func.name,
                                                          self._template_func.args)
        target_stub = copy.deepcopy(self._template_func)
        target_stub.body = "    pass"
        self.task_contract = (
            "# Task\n\n" + evaluation.task_description.strip() +
            "\n\nImplement the target function described below. Its body and any supporting "
            "code are your design space.\n\nTarget function:\n```python\n" +
            str(target_stub).strip() + "\n```"
        )
        self.tree = SearchTree()
        self.rng = random.Random(seed)
        self.parent_selection_counts = {}
        self.step_counter = self.budget_used = self.completed_attempts = 0
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self._invalid_streak = 0
        self.events_path = self.run_dir / "events.jsonl"
        self.llm_calls_path = self.run_dir / "llm_calls.jsonl"
        self.evaluations_path = self.run_dir / "evaluations.jsonl"
        self.pending_path = self.run_dir / "pending_candidate.json"
        self.state_path = self.run_dir / "tree_state.json"
        self.summary_path = self.run_dir / "logs" / "run_summary.json"
        self._outcomes = {r["candidate_id"]: r for r in read_journal(self.evaluations_path)}
        self._logged_calls = {r["call_id"] for r in read_journal(self.llm_calls_path)}
        self._logged_events = {r["candidate_id"] for r in read_journal(self.events_path)}
        events = read_journal(self.events_path)
        self._last_event = events[-1] if events else None
        self._last_response = None
        self.pending = None
        self.mechanism = {
            "method": self.METHOD, "budget": budget, "n_roots": n_roots,
            "parser_protocol": "target_function_anchored_module_v2",
            "traj_gens": traj_gens, "context_margin": context_margin,
            "history_code": history_code,
            "output_tokens": output_tokens, "max_context_tokens": max_context_tokens,
            "operator_probabilities": OPERATOR_PROBABILITIES,
            "quality_ess_target": 8.0, "pivot_uniform_probability": 0.5,
            "donor_uniform_probability": 0.5,
            "task_contract_hash": digest(self.task_contract),
            "llm": {name: getattr(llm, name, None) for name in
                    ("model", "base_url", "temperature", "top_p", "enable_thinking")},
        }
        self.builder = trajectory.TrajectoryBuilder(
            llm, self.task_contract,
            max_tokens=max_context_tokens - context_margin - 1,
            max_events=traj_gens, lookup=self.tree.nodes.get,
            all_nodes=self.tree.all_nodes, include_history_code=history_code,
        )

    def parse_response(self, response, finish_reason="unknown"):
        parsed, error = errors.parse_candidate(response, finish_reason,
                                                self._parse_interface,
                                                self._template_program,
                                                token_counter=self.llm.count_tokens)
        self._parse_error = error
        return parsed

    def _quality_distribution(self, nodes):
        scores = [node.fitness for node in nodes]
        beta, target, quality_ess = calibrate_beta(scores, 8.0)
        maximum = max(scores)
        weights = [math.exp(beta * (node.fitness - maximum)) for node in nodes]
        total = sum(weights)
        return [weight / total for weight in weights], {"quality_ess": quality_ess, "ess_target": target}

    def node_distribution(self, nodes, operator):
        quality, stats = self._quality_distribution(nodes)
        if operator == "Pivot":
            n = len(nodes)
            quality = [0.5 * value + 0.5 / n for value in quality]
        stats["parent_ess"] = ess(quality)
        return quality, stats

    def eligible_nodes(self):
        return self.tree.all_nodes()

    def select_donor(self, parent):
        parent_key = code_key(parent.code)
        nodes = [node for node in self.tree.all_nodes()
                 if node.id != parent.id and code_key(node.code) != parent_key]
        if not nodes:
            return None, []
        quality, _ = self._quality_distribution(nodes)
        n = len(nodes)
        donor = self.rng.choices(nodes, weights=[0.5 * p + 0.5 / n for p in quality])[0]
        return donor, [{"node_id": donor.id, "fitness": donor.fitness}]

    def _schedule(self):
        previous = self._last_event
        if (previous and previous["candidate_id"] == self.completed_attempts and
                previous.get("status") == "eval_failed" and
                previous.get("reason") not in REPAIRABLE_FAILURES):
            raise RuntimeError(f"evaluation infrastructure failed: {previous.get('reason')}")
        if (previous and previous["candidate_id"] == self.completed_attempts and
                (previous.get("status") == "invalid_output" or
                 previous.get("reason") in REPAIRABLE_FAILURES) and
                not previous.get("repair_of")):
            response = self._last_response[1] if self._last_response and self._last_response[0] == previous["candidate_id"] else next(
                record["response"] for record in reversed(read_journal(self.llm_calls_path))
                if record.get("candidate_id") == previous["candidate_id"] and "response" in record)
            prompt = errors.repair_prompt(self.task_contract, response, previous)
            return {"candidate_id": self.completed_attempts + 1, "phase": "selected",
                    "repair_of": previous["candidate_id"], "generation_kind": "repair",
                    "operator": previous.get("operator", "Init"),
                    "requested_operator": previous.get("requested_operator", "Init"),
                    "parent_id": previous.get("parent_id"), "donor_id": previous.get("donor_id"),
                    "parent_fitness": previous.get("parent_fitness"),
                    "donor_fitness": previous.get("donor_fitness"),
                    "best_before": self.tree.best().fitness if self.tree.nodes else None,
                    "prompt": prompt, "prompt_tokens": self.builder.count(prompt, chat=True),
                    "prompt_hash": digest(prompt), "context_best_fitness": previous.get("parent_fitness"),
                    "rng_state": list(self.rng.getstate()), "llm_attempts": 0}
        requested = operator = "Init"
        parent = donor = None
        selection = {}
        if len(self.tree.roots) >= self.n_roots:
            requested = self.rng.choices(OPERATORS, weights=OPERATOR_PROBABILITIES.values())[0]
            operator = requested
            nodes = self.eligible_nodes()
            probabilities, selection = self.node_distribution(nodes, operator)
            index = self.rng.choices(range(len(nodes)), weights=probabilities)[0]
            parent = nodes[index]
            count = self.parent_selection_counts.get(parent.id, 0)
            self.parent_selection_counts[parent.id] = count + 1
            selection.update(parent_probability=probabilities[index], parent_count_before=count)
            if requested == "Fuse":
                donor, donor_attempts = self.select_donor(parent)
                if donor is None:
                    operator = "Refine"
                    selection["fallback_reason"] = "no donor"
        text = self.builder.build_initial() if operator == "Init" else self.builder.build(parent, operator, donor)
        return {"candidate_id": self.completed_attempts + 1, "phase": "selected",
                "requested_operator": requested, "operator": operator,
                "parent_id": parent.id if parent else None, "donor_id": donor.id if donor else None,
                "parent_fitness": parent.fitness if parent else None,
                "donor_fitness": donor.fitness if donor else None,
                "best_before": self.tree.best().fitness if self.tree.nodes else None,
                "selection": selection, "operator_probabilities": OPERATOR_PROBABILITIES,
                "prompt": text, "prompt_tokens": self.builder.count(text, chat=True),
                "prompt_hash": digest(text), "rng_state": list(self.rng.getstate()),
                "llm_attempts": 0}

    def _persist_pending(self):
        atomic_json(self.pending_path, self.pending)

    def _log_call(self, record):
        if record["call_id"] in self._logged_calls:
            return
        self._append_record(self.llm_calls_path, record)
        self._logged_calls.add(record["call_id"])
        if "response" in record:
            self._last_response = (record["candidate_id"], record["response"])

    def _generate_pending(self):
        self.pending["llm_attempts"] += 1
        self._persist_pending()
        started = time.time()
        record = {"ts": datetime.now().isoformat(timespec="seconds"),
                  "call_id": f"{self.pending['candidate_id']}:{self.pending['llm_attempts']}",
                  "candidate_id": self.pending["candidate_id"], "operator": self.pending["operator"],
                  "prompt": self.pending["prompt"], "prompt_tokens": self.pending["prompt_tokens"],
                  "prompt_hash": self.pending["prompt_hash"], "max_tokens": self.output_tokens}
        try:
            details = self.llm.draw_sample_with_details(self.pending["prompt"], max_tokens=self.output_tokens)
        except Exception:
            record.update(seconds=time.time() - started, error=traceback.format_exc())
            self._log_call(record)
            raise
        record.update(response=details["content"], seconds=time.time() - started,
                      finish_reason=details.get("finish_reason") or "unknown", usage=details.get("usage"),
                      model=details.get("model"), response_id=details.get("response_id"),
                      stage="repair" if self.pending.get("repair_of") else "generation")
        self.pending.update(phase="responded", completion=record)
        self._persist_pending()
        self._log_call(record)

    def _evaluate_pending(self, parsed):
        if self.pending["phase"] == "evaluating":
            outcome = self._outcomes.get(self.pending["candidate_id"])
            if outcome is None:
                raise UnknownEvaluation("evaluation reservation has no receipt")
        else:
            self.pending.update(phase="evaluating", evaluation_id=self.budget_used + 1)
            self._persist_pending()
            started = time.time()
            try:
                result = self.secure.evaluate_program_with_details(parsed[2])
                reason, error_type, error, trace = result.failure_kind, result.error_type, result.error, result.traceback
                fitness = None
                if result.result is not None:
                    try:
                        value = float(result.result)
                    except (TypeError, ValueError, OverflowError) as exc:
                        reason, error_type, error = "invalid_result", type(exc).__name__, str(exc)
                    else:
                        if math.isfinite(value):
                            fitness = value
                        else:
                            reason, error_type, error = "nonfinite_fitness", "NonfiniteFitness", "nonfinite fitness"
            except Exception as exc:
                reason, error_type, error, trace = "evaluation_error", type(exc).__name__, str(exc), traceback.format_exc()
                fitness = None
            outcome = {"candidate_id": self.pending["candidate_id"], "evaluation_id": self.pending["evaluation_id"],
                       "fitness": fitness, "reason": reason, "error_type": error_type, "error": error,
                       "traceback": trace, "eval_seconds": time.time() - started,
                       "repair_of": self.pending.get("repair_of")}
            self._append_record(self.evaluations_path, outcome)
            self._outcomes[outcome["candidate_id"]] = outcome
        self.pending.update(phase="evaluated", outcome=outcome)
        self._persist_pending()

    def _advance(self):
        if self.pending is None:
            self.pending = self._schedule()
            self._persist_pending()
        if self.pending["phase"] == "selected":
            self._generate_pending()
        self._log_call(self.pending["completion"])
        completion = self.pending["completion"]
        parsed = self.parse_response(completion["response"], completion["finish_reason"])
        node = None
        if parsed is None:
            self._invalid_streak += 1
            status, reason = "invalid_output", self._parse_error
        else:
            self._invalid_streak = 0
            self._evaluate_pending(parsed)
            outcome = self.pending["outcome"]
            self.budget_used = outcome["evaluation_id"]
            status, reason = ("ok", None) if outcome["fitness"] is not None else ("eval_failed", outcome["reason"])
            if outcome["fitness"] is not None:
                node = self.tree.add(code=parsed[2], idea=parsed[0], fitness=outcome["fitness"],
                                     evaluation_id=self.budget_used, parent_id=self.pending["parent_id"],
                                     operator=self.pending["operator"], donor_id=self.pending["donor_id"])
        if self.pending["parent_id"] is not None:
            self.step_counter += 1
        record = {key: value for key, value in self.pending.items()
                  if key not in ("prompt", "rng_state", "completion", "phase", "outcome")}
        record.update(ts=datetime.now().isoformat(timespec="seconds"), status=status, reason=reason,
                      budget_used=self.budget_used, evaluation_id=self.pending.get("outcome", {}).get("evaluation_id"),
                      eval_seconds=self.pending.get("outcome", {}).get("eval_seconds"),
                      llm_seconds=completion["seconds"], node_id=node.id if node else None,
                      fitness=node.fitness if node else None,
                      idea_tokens=parsed[3] if parsed is not None else None)
        if self.pending.get("outcome") is not None:
            record.update(error_type=self.pending["outcome"].get("error_type"),
                          error=self.pending["outcome"].get("error"))
        if node and self.pending["parent_id"] is not None:
            record.update(parent_improved=node.fitness > self.pending["parent_fitness"],
                          frontier_improved=node.fitness > self.pending["best_before"])
        if self.pending["candidate_id"] not in self._logged_events:
            self._append_record(self.events_path, record)
            self._logged_events.add(self.pending["candidate_id"])
        self.completed_attempts = self.pending["candidate_id"]
        self._save_state()
        self.pending_path.unlink(missing_ok=True)
        self.pending = None
        if self._invalid_streak >= 50:
            raise RuntimeError("50 consecutive generations produced no valid output")

    def _append_record(self, path, record):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path == self.events_path:
            self._last_event = record

    def _save_state(self):
        atomic_json(self.state_path, {"mechanism": self.mechanism,
                                      "started_at": self.started_at, "nodes": [asdict(node) for node in self.tree.all_nodes()],
                                      "rng_state": list(self.rng.getstate()), "parent_selection_counts": self.parent_selection_counts,
                                      "step_counter": self.step_counter, "budget_used": self.budget_used,
                                      "completed_attempts": self.completed_attempts, "invalid_streak": self._invalid_streak})

    def _load_state(self):
        state = json.loads(self.state_path.read_text())
        if state.get("mechanism") != self.mechanism:
            raise ValueError("checkpoint configuration differs from V10.11")
        for entry in state["nodes"]:
            self.tree.add_raw(Node(**entry))
        rng = state["rng_state"]
        self.rng.setstate((rng[0], tuple(rng[1]), rng[2]))
        self.parent_selection_counts = {int(key): value for key, value in state["parent_selection_counts"].items()}
        self.step_counter, self.budget_used = state["step_counter"], state["budget_used"]
        self.completed_attempts, self._invalid_streak = state["completed_attempts"], state["invalid_streak"]
        self.started_at = state["started_at"]
        if self.pending_path.exists():
            pending = json.loads(self.pending_path.read_text())
            if pending["candidate_id"] <= self.completed_attempts:
                self.pending_path.unlink()
            elif pending["candidate_id"] != self.completed_attempts + 1:
                raise ValueError("pending candidate is not the next attempt")
            else:
                self.pending = pending
                rng = pending["rng_state"]
                self.rng.setstate((rng[0], tuple(rng[1]), rng[2]))

    def _write_summary(self, status, error=None):
        best = self.tree.best() if self.tree.nodes else None
        payload = {"status": status, "method": self.METHOD, "started_at": self.started_at,
                   "finished_at": datetime.now().isoformat(timespec="seconds"), "budget": self.budget,
                   "budget_used": self.budget_used, "num_nodes": len(self.tree.nodes),
                   "num_roots": len(self.tree.roots), "num_steps": self.step_counter,
                   "best": None if best is None else {"node_id": best.id, "fitness": best.fitness,
                   "idea": best.idea, "code": best.code, "evaluation_id": best.evaluation_id,
                   "operator": best.operator, "parent_id": best.parent_id, "donor_id": best.donor_id}}
        if error:
            payload["error"] = error
        atomic_json(self.summary_path, payload)

    def run(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            self._load_state()
        else:
            self._save_state()
        try:
            while self.budget_used < self.budget:
                self._advance()
                print(f"{self.METHOD}: budget={self.budget_used}/{self.budget} nodes={len(self.tree.nodes)}", flush=True)
            if len(self.tree.roots) < self.n_roots:
                raise RuntimeError("budget exhausted before initialization completed")
            self._write_summary("finished")
        except UnknownEvaluation:
            self._write_summary("blocked", traceback.format_exc())
            raise
        except KeyboardInterrupt:
            self._write_summary("interrupted")
            raise
        except Exception:
            self._write_summary("error", traceback.format_exc())
            raise

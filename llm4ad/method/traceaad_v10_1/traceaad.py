"""TraceAAD V10.1 search engine.

Mechanism: docs/methods/TraceAAD-V10.1完整机制设计.md — quality-biased
probabilistic parent allocation, three-operator frozen expansion batches,
trajectory-conditioned generation. All five platform tasks maximize, so the
directed quality q(n) equals the raw evaluator fitness.
"""

from __future__ import annotations

import ast
import copy
import json
import math
import random
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

from llm4ad.base import Evaluation, SecureEvaluator, TextFunctionProgramConverter
from llm4ad.base.code import Program

from .prompts import build_prompt, build_task_contract
from .schema import INIT, Node, SearchTree, normalize_code

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
FENCE_RE = re.compile(r"```(?:python)?[ \t]*\r?\n(.*?)```", re.DOTALL)
IDEA_RE = re.compile(r"^[ \t]*Idea:[ \t]*(\S.*?)\s*$", re.MULTILINE)
CHARS_PER_TOKEN = 3.5
MAX_CONSECUTIVE_INVALID = 50


def _strip_thinking(text: str) -> str:
    return THINK_BLOCK_RE.sub("", text)


def _arg_names(args_src: str) -> list[str] | None:
    try:
        tree = ast.parse(f"def _probe_({args_src}):\n    pass")
    except SyntaxError:
        return None
    arguments = tree.body[0].args
    return [a.arg for a in arguments.posonlyargs + arguments.args]


def _ess(beta: float, q_values: list[float]) -> float:
    q_max = max(q_values)
    weights = [math.exp(beta * (q - q_max)) for q in q_values]
    total = sum(weights)
    probs = [w / total for w in weights]
    return 1.0 / sum(p * p for p in probs)


def calibrate_beta(
    q_values: list[float], ess_fraction: float, ess_minimum: int
) -> tuple[float, float, float]:
    """Solve beta >= 0 so that ESS(p) is as close as possible to the target.

    Returns (beta, ess_target, ess_actual). beta = 0 (uniform) whenever the
    uniform distribution already meets the target.
    """
    n = len(q_values)
    target = min(n, max(ess_fraction * n, ess_minimum))
    if n <= 1 or _ess(0.0, q_values) <= target:
        return 0.0, target, _ess(0.0, q_values)
    if max(q_values) == min(q_values):
        # every beta gives the uniform distribution; the target is unreachable
        return 0.0, target, float(n)
    beta_hi = 1.0
    for _ in range(60):
        ess_hi = _ess(beta_hi, q_values)
        if ess_hi <= target:
            break
        if _ess(beta_hi * 10.0, q_values) == ess_hi:
            # ESS saturated at k_max above the target; closest achievable
            return beta_hi, target, ess_hi
        beta_hi *= 10.0
    else:
        return beta_hi, target, _ess(beta_hi, q_values)
    beta_lo = 0.0
    for _ in range(100):
        mid = 0.5 * (beta_lo + beta_hi)
        if _ess(mid, q_values) > target:
            beta_lo = mid
        else:
            beta_hi = mid
    return beta_hi, target, _ess(beta_hi, q_values)


class _Attempt:
    """One scheduled operator's generation-and-evaluation outcome."""

    def __init__(self, operator: str, prompt: str, response: str, llm_seconds: float):
        self.operator = operator
        self.prompt = prompt
        self.response = response
        self.llm_seconds = llm_seconds
        self.idea: str | None = None
        self.code: str | None = None
        self.program: Program | None = None
        self.fitness: float | None = None
        self.reason: str | None = None
        self.eval_seconds: float | None = None
        self.node_id: int | None = None

    @property
    def status(self) -> str:
        if self.program is None:
            return "invalid_output"
        if self.fitness is None:
            return "eval_failed"
        return "ok" if self.node_id is not None else "duplicate"

    @property
    def slot_consumed(self) -> bool:
        return self.program is not None


class TraceAADV101:
    def __init__(
        self,
        *,
        evaluation: Evaluation,
        llm,
        run_dir: Path,
        budget: int = 1000,
        n_roots: int = 8,
        donor_topk: int = 5,
        traj_gens: int = 8,
        ess_fraction: float = 0.1,
        ess_minimum: int = 2,
        output_tokens: int = 16384,
        max_context_tokens: int = 32768,
        seed: int = 0,
    ) -> None:
        if budget < n_roots:
            raise ValueError("budget must cover the initial roots")
        if max_context_tokens <= output_tokens:
            raise ValueError("max_context_tokens must exceed output_tokens")
        self.evaluation = evaluation
        self.llm = llm
        self.run_dir = Path(run_dir)
        self.budget = budget
        self.n_roots = n_roots
        self.donor_topk = donor_topk
        self.traj_gens = traj_gens
        self.ess_fraction = ess_fraction
        self.ess_minimum = ess_minimum
        self.output_tokens = output_tokens
        self.max_context_tokens = max_context_tokens
        self.max_prompt_chars = int(
            (max_context_tokens - output_tokens) * CHARS_PER_TOKEN
        )

        self.events_path = self.run_dir / "events.jsonl"
        self.slots_path = self.run_dir / "slots.jsonl"
        self.batches_path = self.run_dir / "batches.jsonl"
        self.state_path = self.run_dir / "tree_state.json"
        self.summary_path = self.run_dir / "logs" / "run_summary.json"

        self.secure = SecureEvaluator(evaluation)
        self._template = TextFunctionProgramConverter.text_to_program(
            evaluation.template_program
        )
        if self._template is None or len(self._template.functions) != 1:
            raise ValueError("evaluation template must define exactly one function")
        self._template_func = self._template.functions[0]
        self._template_arg_names = _arg_names(self._template_func.args)
        self.task_contract = build_task_contract(
            evaluation.task_description, evaluation.template_program
        )

        self.tree = SearchTree()
        self.rng = random.Random(seed)
        self.batch_counter = 0
        self.budget_used = 0
        self.started_at = datetime.now().isoformat(timespec="seconds")
        self._invalid_streak = 0

    # ------------------------------------------------------------------
    # response parsing
    # ------------------------------------------------------------------

    def parse_response(self, response: str) -> tuple[str, str, Program] | None:
        """Parse Idea + Code from a response into (idea, canonical code,
        executable program), or None when the output violates the contract
        (no evaluator slot is consumed in that case)."""
        text = _strip_thinking(response)
        match = IDEA_RE.search(text)
        if match is None:
            return None
        fence = FENCE_RE.search(text)
        block = fence.group(1) if fence else text
        program = TextFunctionProgramConverter.text_to_program(block)
        if program is None or not program.functions:
            return None
        named = [f for f in program.functions if f.name == self._template_func.name]
        if named:
            target = named[0]
        elif len(program.functions) == 1:
            target = program.functions[0]
        else:
            return None
        if _arg_names(target.args) != self._template_arg_names:
            return None
        canonical = normalize_code(block)
        if not canonical:
            return None
        extra = program.preface.strip()
        for func in program.functions:
            if func is not target:
                extra = (extra + "\n\n" + str(func).rstrip()).strip()
        template_func = copy.deepcopy(self._template_func)
        template_func.body = target.body
        preface = self._template.preface.strip()
        if extra:
            preface = f"{preface}\n\n{extra}" if preface else extra
        return (
            match.group(1).strip(),
            canonical,
            Program(preface=preface, functions=[template_func]),
        )

    # ------------------------------------------------------------------
    # allocation
    # ------------------------------------------------------------------

    def select_parent(self) -> tuple[Node, float, float, float, float]:
        """Sample one parent from the ESS-calibrated Boltzmann distribution."""
        nodes = self.tree.all_nodes()
        q_values = [n.fitness for n in nodes]
        beta, ess_target, ess_actual = calibrate_beta(
            q_values, self.ess_fraction, self.ess_minimum
        )
        q_max = max(q_values)
        weights = [math.exp(beta * (q - q_max)) for q in q_values]
        total = sum(weights)
        probs = [w / total for w in weights]
        index = self.rng.choices(range(len(nodes)), weights=probs)[0]
        return nodes[index], probs[index], beta, ess_target, ess_actual

    def select_donor(self, parent: Node) -> Node | None:
        """Uniform pick from the top-5 cross-lineage nodes by current quality."""
        excluded = {parent.id} | {n.id for n in self.tree.ancestors(parent.id)}
        excluded |= self.tree.descendants(parent.id)
        candidates = [n for n in self.tree.all_nodes() if n.id not in excluded]
        candidates.sort(key=lambda n: (-n.fitness, n.id))
        top = candidates[: self.donor_topk]
        return self.rng.choice(top) if top else None

    # ------------------------------------------------------------------
    # logging and state
    # ------------------------------------------------------------------

    def _append_jsonl(self, path: Path, payload: dict) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _log_attempt(self, attempt: _Attempt, *, phase: str, batch: int,
                     parent_id: int | None, donor_id: int | None) -> None:
        self._append_jsonl(
            self.events_path,
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "phase": phase,
                "batch": batch,
                "operator": attempt.operator,
                "parent_id": parent_id,
                "donor_id": donor_id,
                "status": attempt.status,
                "reason": attempt.reason,
                "idea": attempt.idea,
                "code": attempt.code,
                "fitness": attempt.fitness,
                "node_id": attempt.node_id,
                "slot_consumed": attempt.slot_consumed,
                "llm_seconds": round(attempt.llm_seconds, 3),
                "eval_seconds": None
                if attempt.eval_seconds is None
                else round(attempt.eval_seconds, 3),
                "prompt": attempt.prompt,
                "response": attempt.response,
            },
        )

    def _save_state(self) -> None:
        state = {
            "version": 1,
            "started_at": self.started_at,
            "nodes": self.tree.to_state(),
            "rng_state": list(self.rng.getstate()),
            "batch_counter": self.batch_counter,
            "budget_used": self.budget_used,
        }
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)

    def _load_state(self) -> None:
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        for entry in state["nodes"]:
            self.tree.add_raw(Node(**entry))
        rng_state = state["rng_state"]
        self.rng.setstate((rng_state[0], tuple(rng_state[1]), rng_state[2]))
        self.batch_counter = state["batch_counter"]
        self.started_at = state["started_at"]
        # budget truth: one line per completed formal evaluation, written
        # immediately after the evaluator returns
        self.budget_used = 0
        if self.slots_path.exists():
            with self.slots_path.open(encoding="utf-8") as handle:
                self.budget_used = sum(1 for _ in handle)

    def _write_summary(self, status: str, error: str | None = None) -> None:
        best = self.tree.best() if self.tree.nodes else None
        payload = {
            "status": status,
            "method": "v101",
            "started_at": self.started_at,
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "budget": self.budget,
            "budget_used": self.budget_used,
            "num_nodes": len(self.tree.nodes),
            "num_roots": len(self.tree.roots),
            "num_batches": self.batch_counter,
            "best": None
            if best is None
            else {
                "node_id": best.id,
                "fitness": best.fitness,
                "idea": best.idea,
                "code": best.code,
                "origin_operator": best.origin_operator,
                "parent_id": best.parent_id,
                "donor_id": best.donor_id,
            },
        }
        if error is not None:
            payload["error"] = error
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # ------------------------------------------------------------------
    # search phases
    # ------------------------------------------------------------------

    def _prompt(
        self, current: Node | None, ancestors: list[Node], operator: str, donor: Node | None
    ) -> str:
        return build_prompt(
            task_contract=self.task_contract,
            current=current,
            ancestors=ancestors,
            operator=operator,
            donor=donor,
            max_prompt_chars=self.max_prompt_chars,
            max_gens=self.traj_gens,
        )

    def _generate(self, operator: str, current: Node | None,
                  ancestors: list[Node], donor: Node | None) -> _Attempt:
        prompt = self._prompt(current, ancestors, operator, donor)
        start = time.time()
        response = self.llm.draw_sample(prompt)
        attempt = _Attempt(operator, prompt, response, time.time() - start)
        parsed = self.parse_response(response)
        if parsed is not None:
            attempt.idea, attempt.code, attempt.program = parsed
            self._invalid_streak = 0
        else:
            attempt.reason = "idea missing or code unparseable/signature mismatch"
            self._invalid_streak += 1
            if self._invalid_streak > MAX_CONSECUTIVE_INVALID:
                raise RuntimeError(
                    f"{MAX_CONSECUTIVE_INVALID} consecutive generations produced "
                    "no valid output; the serving backend is misbehaving"
                )
        return attempt

    def _evaluate(self, attempt: _Attempt, *, phase: str, batch: int) -> None:
        start = time.time()
        outcome = self.secure.evaluate_program_with_details(attempt.program)
        attempt.eval_seconds = time.time() - start
        self.budget_used += 1
        # slot evidence hits the ledger before anything else can fail
        self._append_jsonl(
            self.slots_path,
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "phase": phase,
                "batch": batch,
                "operator": attempt.operator,
            },
        )
        if outcome.result is None:
            attempt.reason = outcome.failure_kind
        else:
            attempt.fitness = float(outcome.result)

    def _initialize(self) -> None:
        while len(self.tree.roots) < self.n_roots:
            if self.budget_used >= self.budget:
                raise RuntimeError(
                    f"budget exhausted during initialization: "
                    f"{len(self.tree.roots)}/{self.n_roots} valid roots"
                )
            attempt = self._generate(INIT, None, [], None)
            if attempt.program is not None:
                self._evaluate(attempt, phase="init", batch=0)
                if attempt.fitness is not None:
                    node = self.tree.add(
                        code=attempt.code,
                        idea=attempt.idea,
                        fitness=attempt.fitness,
                        parent_id=None,
                        origin_operator=INIT,
                    )
                    if node is not None:
                        attempt.node_id = node.id
            self._log_attempt(attempt, phase="init", batch=0,
                              parent_id=None, donor_id=None)
            self._save_state()

    def _run_batch(self) -> None:
        remaining = self.budget - self.budget_used
        budget_before = self.budget_used
        nodes_before = len(self.tree.nodes)
        parent, parent_prob, beta, ess_target, ess_actual = self.select_parent()
        # one extra oldest ancestor beyond the display window, for trends only
        ancestors = self.tree.ancestors(parent.id)[: self.traj_gens + 1]
        donor = self.select_donor(parent)
        operators = ["Refine", "Pivot", "Fuse"] if donor is not None else ["Refine", "Pivot"]
        if remaining < len(operators):
            operators = self.rng.sample(operators, remaining)

        # phase 1: every scheduled operator generates from the same frozen
        # snapshot, before any sibling is evaluated; only Fuse sees the donor
        attempts = [
            self._generate(
                operator, parent, ancestors, donor if operator == "Fuse" else None
            )
            for operator in operators
        ]

        # phase 2: independent formal evaluations
        for attempt in attempts:
            if attempt.program is not None:
                self._evaluate(attempt, phase="batch", batch=self.batch_counter + 1)

        # commit: log attempts, add valid non-duplicate children
        for attempt in attempts:
            if attempt.program is not None and attempt.fitness is not None:
                node = self.tree.add(
                    code=attempt.code,
                    idea=attempt.idea,
                    fitness=attempt.fitness,
                    parent_id=parent.id,
                    origin_operator=attempt.operator,
                    donor_id=donor.id if attempt.operator == "Fuse" else None,
                )
                if node is not None:
                    attempt.node_id = node.id
            self._log_attempt(
                attempt,
                phase="batch",
                batch=self.batch_counter + 1,
                parent_id=parent.id,
                donor_id=donor.id if attempt.operator == "Fuse" else None,
            )

        self.batch_counter += 1
        self._append_jsonl(
            self.batches_path,
            {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "batch": self.batch_counter,
                "budget_before": budget_before,
                "budget_after": self.budget_used,
                "n_candidates": nodes_before,
                "beta": beta,
                "ess_target": ess_target,
                "ess_actual": ess_actual,
                "parent_id": parent.id,
                "parent_fitness": parent.fitness,
                "parent_probability": parent_prob,
                "donor_id": None if donor is None else donor.id,
                "operators": operators,
                "outcomes": {
                    a.operator: {
                        "status": a.status,
                        "fitness": a.fitness,
                        "node_id": a.node_id,
                    }
                    for a in attempts
                },
                "nodes_after": len(self.tree.nodes),
            },
        )
        self._save_state()

    def run(self) -> None:
        if self.state_path.exists():
            self._load_state()
            print(
                f"resumed: nodes={len(self.tree.nodes)} roots={len(self.tree.roots)} "
                f"batches={self.batch_counter} budget_used={self.budget_used}",
                flush=True,
            )
        try:
            if len(self.tree.roots) < self.n_roots:
                self._initialize()
                print(
                    f"initialization done: {len(self.tree.roots)} roots, "
                    f"budget_used={self.budget_used}",
                    flush=True,
                )
            while self.budget_used < self.budget:
                self._run_batch()
                print(
                    f"batch {self.batch_counter}: budget {self.budget_used}/{self.budget} "
                    f"nodes={len(self.tree.nodes)} best={self.tree.best().fitness}",
                    flush=True,
                )
            self._write_summary("finished")
            print(
                f"finished: budget_used={self.budget_used} "
                f"best={self.tree.best().fitness}",
                flush=True,
            )
        except KeyboardInterrupt:
            self._write_summary("interrupted")
            raise
        except Exception:
            self._write_summary("error", error=traceback.format_exc())
            raise

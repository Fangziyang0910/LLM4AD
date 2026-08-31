from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v10 import RunArtifacts, TraceAADV10
from llm4ad.method.traceaad_v10.critic import (
    build_critic_prompt,
    fallback_result,
    parse_critic_response,
)
from llm4ad.method.traceaad_v10.traceaad import CandidateOutcome
from llm4ad.method.traceaad_v10.opportunity import (
    build_opportunities,
    coverage_tuple,
    mid_rank,
    operator_observations,
    reference_candidates,
    screening_index,
    screen_shortlist,
    select_by_coverage,
)
from llm4ad.method.traceaad_v10.prompt import (
    OPERATOR_INSTRUCTIONS,
    build_generation_prompt,
    parse_candidate_response,
)
from llm4ad.method.traceaad_v10.schema import (
    OPENING_OPERATORS,
    SEMANTIC_REPAIR,
    AttemptRecord,
    CompetitiveEntry,
    Opportunity,
    Pending,
    ProgramNode,
    Thread,
)


class ToyEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program="def f(x):\n    return 0.0\n",
            task_description="Return a high scalar.",
            safe_evaluate=False,
            timeout_seconds=5,
        )

    def evaluate_program(self, program_str: str, callable_func, **kwargs):
        return float(callable_func(1.0))


class ToyLLM(LLM):
    """Roots and generations return increasing fitness; the critic echoes the
    first four opportunity ids as a valid competitive set."""

    def __init__(self, *, critic_junk: bool = False) -> None:
        super().__init__()
        self.calls = 0
        self.critic_calls = 0
        self.critic_junk = critic_junk
        self.critic_prompts: list[str] = []
        self.generation_prompts: list[str] = []

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        if "valuation critic" in prompt:
            self.critic_calls += 1
            self.critic_prompts.append(prompt)
            if self.critic_junk:
                return "I cannot judge this."
            lines = re.findall(r"^(O\d+): (.*)$", prompt, flags=re.MULTILINE)
            entries = []
            for rank, (opportunity_id, description) in enumerate(lines[:4], 1):
                entry = {
                    "opportunity_id": opportunity_id,
                    "rank": rank,
                    "reason": f"reason {rank}",
                    "evidence_refs": [],
                    "expected_payoff_horizon": "short",
                    "semantic_mismatch": None,
                }
                if "semantic_repair" in description:
                    entry["semantic_mismatch"] = "the stated ranking idea is not applied in the loop"
                entries.append(entry)
            return "```json\n" + json.dumps({"competitive_set": entries}) + "\n```"
        self.calls += 1
        self.generation_prompts.append(prompt)
        value = self.calls / 100.0
        return (
            f"Idea: direction {self.calls}\nCode:\n```python\n"
            f"def f(x):\n    return {value}\n```"
        )


def _make_method(
    tmp_path: Path,
    llm: LLM,
    *,
    budget: int = 6,
    n_roots: int = 2,
) -> TraceAADV10:
    run_dir = tmp_path / "run"
    return TraceAADV10(
        llm=llm,
        evaluation=ToyEvaluation(),
        artifacts=RunArtifacts(run_dir),
        budget=budget,
        n_roots=n_roots,
        seed=0,
        checkpoint_dir=run_dir / "checkpoints",
        task_key="toy",
    )


def _pending(
    *,
    operator: str,
    slot: int,
    start_id: int | None,
    start_fitness: float | None,
    q_origin: float | None = None,
    idea: str | None = None,
) -> Pending:
    return Pending(
        prompt="",
        response="",
        operator=operator,
        idea=idea or f"{operator} idea",
        slot=slot,
        round_index=slot,
        start_id=start_id,
        reference_id=None,
        base_code="",
        start_fitness=start_fitness,
        q_origin=q_origin,
        semantic_mismatch=None,
        opportunity_id="O0",
        critic_rank=1,
    )


# ----------------------------------------------------------------------
# End-to-end toy run
# ----------------------------------------------------------------------


def test_v10_runs_one_experiment_per_slot(tmp_path: Path) -> None:
    llm = ToyLLM()
    method = _make_method(tmp_path, llm, budget=6, n_roots=2)
    method.run()

    run_dir = tmp_path / "run"
    assert method._n_eval == 6
    assert method._n_calls == 6
    rows = list(csv.DictReader((run_dir / "evaluations.csv").open(newline="")))
    assert len(rows) == 6
    assert [row["operator"] for row in rows[:2]] == ["root", "root"]
    design_rows = rows[2:]
    assert {row["operator"] for row in design_rows} <= {
        "develop",
        "pivot",
        "transfer",
        "restart",
        "semantic_repair",
    }
    # every non-init thread was opened by a recorded opening attempt whose
    # start fitness is exactly the thread's origin quality
    opening_rows = [row for row in design_rows if row["operator"] in {"pivot", "transfer", "restart"}]
    init_threads = [t for t in method._threads.values() if t.origin_action == "init"]
    opened_threads = [t for t in method._threads.values() if t.origin_action != "init"]
    assert len(init_threads) == 2
    assert all(thread.q_origin is None for thread in init_threads)
    assert len(opened_threads) == len(
        [row for row in opening_rows if row["outcome"] in {"improve", "plateau", "regress"}]
    )
    for thread in opened_threads:
        opening = next(
            row
            for row in opening_rows
            if row["created_thread"] == str(thread.id)
        )
        if thread.origin_action == "restart":
            assert float(opening["q_origin"]) == pytest.approx(thread.q_origin)
        else:
            assert float(opening["start_fitness"]) == pytest.approx(thread.q_origin)

    decisions = [
        json.loads(line)
        for line in (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    critic_decisions = [item for item in decisions if item["stage"] == "critic"]
    generation_decisions = [item for item in decisions if item["stage"] == "generation"]
    root_decisions = [item for item in decisions if item["stage"] == "root"]
    assert len(critic_decisions) == 4  # one critic call per primary slot
    assert len(generation_decisions) == 4  # four experiments
    assert len(root_decisions) == 2  # two initializations
    for decision in critic_decisions:
        assert len(decision["competitive_set"]) == 4
        assert decision["invalid"] is False

    events = [
        json.loads(line)
        for line in (run_dir / "mechanism_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rounds = [item for item in events if item["event"] == "round_start"]
    allocations = [item for item in events if item["event"] == "allocation"]
    settles = [item for item in events if item["event"] == "settle"]
    design_settles = [item for item in settles if item["operator"] != "root"]
    assert len(settles) == 6  # every primary slot settles exactly once
    assert len(rounds) == len(allocations) == len(design_settles) == 4
    assert [len(event["shortlist"]) for event in rounds] == [2, 3, 4, 5]  # archive grows one node per slot
    for event in rounds:
        items = event["opportunities"]
        ids = {item["opportunity_id"] for item in items}
        # every start x (develop, pivot, >=1 transfer, semantic_repair) + exactly one restart
        assert len(ids) >= len(event["shortlist"]) * 4 + 1
        assert sum(1 for item in items if item["operator"] == "restart") == 1
        for start in event["shortlist"]:
            operators = {
                item["operator"]
                for item in items
                if item["start_id"] == start["id"]
            }
            assert {"develop", "pivot", "transfer", "semantic_repair"} <= operators

    summary = json.loads((run_dir / "logs" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "finished"
    assert summary["budget_slots"] == 6
    assert summary["mechanism"] == "trajectory_aware_joint_design_opportunity_allocation"
    assert summary["critic_llm_calls"] == 4
    assert summary["critic_invalid"] == 0
    assert summary["constants"] == {
        "K_s": 8,
        "K_d": 2,
        "K_c": 4,
        "H_tau": 8,
        "H_G": [1, 2, 4],
        "N_root": 2,
        "N_card": 3,
        "max_repairs": 2,
    }
    best_rows = [
        json.loads(line)
        for line in (run_dir / "best_history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert best_rows
    assert max(item["fitness"] for item in best_rows) == pytest.approx(method.best.fitness)


def test_v10_critic_fallback_counts_invalid(tmp_path: Path) -> None:
    llm = ToyLLM(critic_junk=True)
    method = _make_method(tmp_path, llm, budget=3, n_roots=2)
    method.run()

    assert method._critic_llm_calls == 2  # one slot, retried once
    assert method._critic_invalid == 1
    decisions = [
        json.loads(line)
        for line in (tmp_path / "run" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    critic = [item for item in decisions if item["stage"] == "critic"][0]
    assert critic["invalid"] is True
    assert [entry["operator"] for entry in critic["competitive_set"]] == ["develop", "develop"]
    assert len(critic["competitive_set"]) == 2  # only two shortlist starts exist
    settles = [
        json.loads(line)
        for line in (tmp_path / "run" / "mechanism_events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line)["event"] == "settle"
    ]
    assert settles[-1]["operator"] == "develop"  # the fallback allocates Develop


def test_v10_checkpoint_roundtrip_preserves_state(tmp_path: Path) -> None:
    first = _make_method(tmp_path, ToyLLM(), budget=5, n_roots=2)
    first.run()
    checkpoint = tmp_path / "run" / "checkpoints" / "latest.json"
    resumed = TraceAADV10(
        ToyLLM(),
        ToyEvaluation(),
        budget=5,
        n_roots=2,
        checkpoint_dir=tmp_path / "resumed" / "checkpoints",
        resume_from=checkpoint,
        task_key="toy",
    )

    assert resumed._n_eval == first._n_eval
    assert resumed._n_calls == first._n_calls
    assert resumed._pending is None
    assert {key: as_list(value) for key, value in resumed._threads.items()} == {
        key: as_list(value) for key, value in first._threads.items()
    }
    assert resumed._attempts == first._attempts
    assert resumed._llm_tokens == first._llm_tokens
    assert resumed.best is not None
    assert resumed.best.fitness == pytest.approx(first.best.fitness)


def as_list(thread: Thread) -> list:
    return [thread.origin_action, thread.q_origin, thread.opportunities_used, thread.best_history]


# ----------------------------------------------------------------------
# Screening and mid-rank
# ----------------------------------------------------------------------


def test_v10_mid_rank_matches_specification() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    assert mid_rank(4.0, values) == pytest.approx(1.0)
    assert mid_rank(1.0, values) == pytest.approx(0.0)
    assert mid_rank(1.0, [1.0]) == pytest.approx(0.5)
    assert mid_rank(2.0, [2.0, 2.0, 2.0]) == pytest.approx(0.5)
    # tie between two nodes shares the middle ranks
    assert mid_rank(2.0, [1.0, 2.0, 2.0, 4.0]) == pytest.approx((1 + 0.5) / 3)


def test_v10_screening_index_rewards_underchecked_states() -> None:
    assert screening_index(0.5, 0) == pytest.approx(1.5)
    assert screening_index(0.5, 3) < screening_index(0.5, 0)
    assert screening_index(1.0, 100) > screening_index(0.0, 0)


def _node(node_id: int, fitness: float, *, parent: int | None = None, thread: int = 1) -> ProgramNode:
    return ProgramNode(node_id, f"code-{node_id}", fitness, parent, thread, f"idea {node_id}", node_id)


def test_v10_screen_shortlist_replaces_lowest_with_incumbent() -> None:
    # 12 nodes: the incumbent is heavily used, nine barely-worse nodes are unused.
    nodes = {1: _node(1, 100.0)}
    for i in range(2, 13):
        nodes[i] = _node(i, 99.0)
    attempts = [
        AttemptRecord(
            slot=slot,
            round_index=slot,
            operator="develop",
            idea="i",
            outcome="regress",
            start_id=1,
            start_fitness=100.0,
            child_id=None,
            child_fitness=None,
            thread_of_start=1,
            created_thread=None,
        )
        for slot in range(1, 16)
    ]
    assert len(nodes) == 12
    without = screen_shortlist(
        nodes, attempts, best_node_id=1, size=8, rng=random.Random(0)
    )
    assert len(without) == 8
    assert 1 in without  # incumbent kept despite heavy use


def test_v10_screen_shortlist_prefers_underchecked_states() -> None:
    nodes = {i: _node(i, float(i)) for i in range(1, 12)}  # node 11 is the incumbent
    attempts = [
        AttemptRecord(
            slot=slot,
            round_index=slot,
            operator="develop",
            idea="i",
            outcome="regress",
            start_id=start,
            start_fitness=1.0,
            child_id=None,
            child_fitness=None,
            thread_of_start=1,
            created_thread=None,
        )
        for slot, start in [
            (1, 11), (2, 11), (3, 11), (4, 11), (5, 11),
            (6, 10), (7, 10), (8, 10), (9, 10), (10, 10),
        ]
    ]
    shortlist = screen_shortlist(nodes, attempts, best_node_id=11, size=3, rng=random.Random(0))
    assert shortlist == [8, 9, 11]  # unused next-best states screen first, incumbent replaces the weakest


def test_v10_reference_candidates_exclude_same_thread_and_lineage() -> None:
    nodes = {
        1: _node(1, 10.0, thread=1),
        2: _node(2, 9.0, parent=1, thread=1),   # same thread, ancestor line
        3: _node(3, 8.0, parent=2, thread=1),   # descendant of 2
        4: _node(4, 7.0, thread=2),
        5: _node(5, 6.0, parent=4, thread=2),
    }
    candidates = reference_candidates(nodes, nodes[3])
    assert candidates == [4, 5]  # other thread, best quality first


def test_v10_opportunity_set_shape() -> None:
    nodes = {
        1: _node(1, 10.0, thread=1),
        2: _node(2, 9.0, thread=2),
        3: _node(3, 8.0, parent=1, thread=1),
    }
    opportunities, references = build_opportunities(nodes, [1, 3])
    by_start: dict[int, list[str]] = {}
    for opportunity in opportunities:
        if opportunity.start_id is not None:
            by_start.setdefault(opportunity.start_id, []).append(opportunity.operator)
    assert set(by_start[1]) == {"develop", "pivot", "transfer", "semantic_repair"}
    assert set(by_start[3]) == {"develop", "pivot", "transfer", "semantic_repair"}
    restarts = [o for o in opportunities if o.operator == "restart"]
    assert len(restarts) == 1 and restarts[0].start_id is None
    assert references[3] == [2]  # node 2 is the only other-thread candidate
    transfers = [o for o in opportunities if o.operator == "transfer"]
    assert {o.reference_id for o in transfers} == {2}


# ----------------------------------------------------------------------
# Threads, G_h, and operator observations
# ----------------------------------------------------------------------


def test_v10_thread_bookkeeping_and_g_values(tmp_path: Path) -> None:
    method = TraceAADV10(ToyLLM(), ToyEvaluation(), budget=20, n_roots=2, task_key="toy")

    def settle(pending: Pending, fitness: float, outcome: str) -> None:
        method._pending = pending
        node, created = method._register_valid_child(pending, f"code-{fitness}", fitness)
        method._finish_attempt(
            CandidateOutcome(
                code=f"code-{fitness}",
                fitness=fitness,
                outcome=outcome,
                node_id=node.id,
                created_thread=created,
                error=None,
                error_type=None,
                attempt=1,
            )
        )

    settle(_pending(operator="root", slot=1, start_id=None, start_fitness=None), 10.0, "new_root")
    settle(_pending(operator="root", slot=2, start_id=None, start_fitness=None), 8.0, "new_root")
    settle(_pending(operator="pivot", slot=3, start_id=2, start_fitness=8.0), 5.0, "regress")

    thread_one, thread_two, thread_three = (method._threads[i] for i in (1, 2, 3))
    assert thread_one.origin_action == "init" and thread_one.q_origin is None
    assert thread_two.opportunities_used == 2  # its own root plus the pivot from it
    assert thread_two.best_history == [8.0, 8.0]  # the pivot child joined a new thread
    assert thread_three.origin_action == "pivot"
    assert thread_three.q_origin == 8.0
    assert thread_three.opportunities_used == 1
    assert thread_three.g_value(1) == pytest.approx(-3.0)  # first step dropped
    assert thread_three.g_value(2) is None  # not yet observed, never zero-filled
    assert method._nodes[3].thread_id == 3

    settle(_pending(operator="develop", slot=4, start_id=3, start_fitness=5.0), 12.0, "improve")
    assert thread_three.opportunities_used == 2
    assert thread_three.best_history == [5.0, 12.0]
    assert thread_three.g_value(2) == pytest.approx(4.0)  # delayed payoff realized
    assert method._nodes[4].thread_id == 3  # continuation joins the opening thread

    observations = operator_observations(method._attempts, method._threads)
    pivot_stats = observations["pivot"]
    assert pivot_stats["P_G2_positive"] == pytest.approx(1.0)
    assert pivot_stats["dropped_threads"] == 1
    assert pivot_stats["recovered_threads"] == 1
    assert pivot_stats["mean_recovery_slots"] == pytest.approx(2)
    develop_stats = observations["develop"]
    assert develop_stats["trials"] == 1
    assert develop_stats["one_step_improvement_rate"] == pytest.approx(1.0)
    assert "P_G2_positive" not in develop_stats  # opening operators only


def test_v10_failed_opening_consumes_start_thread_budget(tmp_path: Path) -> None:
    method = TraceAADV10(ToyLLM(), ToyEvaluation(), budget=20, n_roots=1, task_key="toy")
    method._pending = _pending(operator="root", slot=1, start_id=None, start_fitness=None)
    method._register_valid_child(method._pending, "code-root", 10.0)
    method._finish_attempt(
        CandidateOutcome("code-root", 10.0, "new_root", 1, 1, None, None, 1)
    )
    pending = _pending(operator="pivot", slot=2, start_id=1, start_fitness=10.0)
    method._pending = pending
    method._consume_start_thread_slot(pending, joined_fitness=None)
    method._finish_attempt(
        CandidateOutcome("broken", None, "invalid", None, None, "boom", "SyntaxError", 3)
    )
    thread = method._threads[1]
    assert thread.opportunities_used == 2  # root creation + failed pivot
    assert thread.best_history == [10.0, 10.0]
    assert len(method._threads) == 1  # failed openings create no thread


def test_v10_restart_thread_uses_global_best_origin(tmp_path: Path) -> None:
    method = TraceAADV10(ToyLLM(), ToyEvaluation(), budget=20, n_roots=1, task_key="toy")
    method._pending = _pending(operator="root", slot=1, start_id=None, start_fitness=None)
    method._register_valid_child(method._pending, "code-root", 10.0)
    method._finish_attempt(
        CandidateOutcome("code-root", 10.0, "new_root", 1, 1, None, None, 1)
    )
    pending = _pending(operator="restart", slot=2, start_id=None,
                       start_fitness=None, q_origin=10.0)
    method._pending = pending
    node, created = method._register_valid_child(pending, "code-restart", 4.0)
    method._finish_attempt(
        CandidateOutcome("code-restart", 4.0, "regress", node.id, created, None, None, 1)
    )
    assert node.parent_id is None  # a restart is not anchored to any parent
    thread = method._threads[created]
    assert thread.origin_action == "restart"
    assert thread.q_origin == 10.0
    assert thread.g_value(1) == pytest.approx(-6.0)
    restart_stats = operator_observations(method._attempts, method._threads)["restart"]
    assert restart_stats["trials"] == 1
    assert restart_stats["valid_rate"] == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Coverage allocation
# ----------------------------------------------------------------------


def _attempt(operator: str, start_id: int | None, *, thread: int | None = None) -> AttemptRecord:
    return AttemptRecord(
        slot=1,
        round_index=1,
        operator=operator,
        idea="i",
        outcome="regress",
        start_id=start_id,
        start_fitness=None,
        child_id=None,
        child_fitness=None,
        thread_of_start=thread,
        created_thread=None,
    )


def test_v10_coverage_tuple_lexicographic_semantics() -> None:
    nodes = {1: _node(1, 1.0, thread=10), 2: _node(2, 2.0, thread=10)}
    attempts = [
        _attempt("develop", 1, thread=10),
        _attempt("develop", 1, thread=10),
        _attempt("develop", 2, thread=10),
        _attempt("pivot", 1, thread=10),
        _attempt("transfer", 1, thread=10),
    ]
    develop_one = Opportunity("O1", "develop", 1, None)
    develop_two = Opportunity("O2", "develop", 2, None)
    pivot_one = Opportunity("O3", "pivot", 1, None)
    transfer_two = Opportunity("O4", "transfer", 2, None)
    assert coverage_tuple(develop_one, attempts, nodes) == (2, 3, 3)
    assert coverage_tuple(develop_two, attempts, nodes) == (1, 3, 3)
    assert coverage_tuple(pivot_one, attempts, nodes) == (1, 1, 1)
    assert coverage_tuple(transfer_two, attempts, nodes) == (0, 1, 1)
    rng = random.Random(0)
    winner = select_by_coverage(
        [
            _entry(develop_one, rank=1),
            _entry(develop_two, rank=2),
            _entry(pivot_one, rank=3),
        ],
        attempts,
        nodes,
        rng,
    )
    assert winner.opportunity.opportunity_id == "O3"  # lexicographic minimum first


def _entry(opportunity: Opportunity, rank: int) -> CompetitiveEntry:
    return CompetitiveEntry(
        opportunity=opportunity,
        rank=rank,
        reason="r",
        evidence_refs=(),
        payoff_horizon=None,
        semantic_mismatch=None,
    )


def test_v10_restart_coverage_thread_element_is_zero() -> None:
    nodes = {1: _node(1, 1.0, thread=10)}
    attempts = [_attempt("restart", None), _attempt("restart", None)]
    restart = Opportunity("O9", "restart", None, None)
    assert coverage_tuple(restart, attempts, nodes) == (2, 0, 2)


def test_v10_transfer_coverage_ignores_reference_choice() -> None:
    nodes = {1: _node(1, 1.0, thread=10), 2: _node(2, 2.0, thread=20), 3: _node(3, 3.0, thread=30)}
    attempts = [_attempt("transfer", 1, thread=10)]
    first = Opportunity("O1", "transfer", 1, 2)
    second = Opportunity("O2", "transfer", 1, 3)
    assert coverage_tuple(first, attempts, nodes) == coverage_tuple(second, attempts, nodes)


def test_v10_ties_break_by_rank_then_seed() -> None:
    nodes = {1: _node(1, 1.0)}
    opportunity_one = Opportunity("O1", "develop", 1, None)
    opportunity_two = Opportunity("O2", "pivot", 1, None)
    winner = select_by_coverage(
        [_entry(opportunity_two, rank=2), _entry(opportunity_one, rank=1)],
        [],
        nodes,
        random.Random(0),
    )
    assert winner.opportunity.opportunity_id == "O1"


# ----------------------------------------------------------------------
# Critic prompt and response validation
# ----------------------------------------------------------------------


def _critic_fixture() -> tuple[dict, dict, list, dict]:
    nodes = {
        1: _node(1, 10.0, thread=1),
        2: _node(2, 8.0, parent=1, thread=1),
        3: _node(3, 9.0, thread=2),
    }
    threads = {
        1: Thread(1, "init", "root idea", 1, 1, None, 2, [8.0, 10.0]),
        2: Thread(2, "pivot", "pivot idea", 2, 3, 9.0, 1, [9.0]),
    }
    attempts = [
        AttemptRecord(2, 1, "develop", "idea a", "regress", 1, 10.0, 2, 8.0, 1, None),
        AttemptRecord(3, 1, "pivot", "idea b", "regress", 1, 10.0, 3, 9.0, 1, 2),
    ]
    opportunities, references = build_opportunities(nodes, [2])
    return nodes, threads, (opportunities, references), attempts


def _built_prompt():
    nodes, threads, (opportunities, references), attempts = _critic_fixture()
    built = build_critic_prompt(
        task_description="Return a high scalar.",
        slot=4,
        remaining_budget=17,
        primary_evaluations=3,
        best_node=nodes[1],
        nodes=nodes,
        threads=threads,
        attempts=attempts,
        shortlist=[2],
        opportunities=opportunities,
        references=references,
    )
    return built, nodes, opportunities


def test_v10_critic_prompt_shows_labels_paths_ledger_and_g() -> None:
    built, nodes, opportunities = _built_prompt()
    prompt = built.prompt
    assert "Remaining budget: 17" in prompt
    assert "Global best fitness: 10 (node S1)" in prompt
    assert "[OP-develop]" in prompt and "[OP-restart]" in prompt
    assert "P_G2_positive" in prompt
    assert "S2-G" in prompt  # thread G summary label
    assert "[S2-H1] regress" in prompt  # labeled formation step
    assert "[S2-Ldevelop]" in prompt  # labeled ledger summary
    assert "S2-R3" in prompt  # labeled transfer reference
    assert "origin_idea=root idea" in prompt  # start thread summary
    assert "thread origin idea: pivot idea" in prompt  # reference thread summary
    # the reference node 3 code appears once, the start code appears once
    assert prompt.count("code-3") == 1
    for opportunity in opportunities:
        assert f"{opportunity.opportunity_id}:" in prompt
    assert "S2-H1" in built.valid_labels
    assert "OP-develop" in built.valid_labels
    assert "S2-Lpivot" in built.valid_labels


def test_v10_critic_parse_accepts_valid_response() -> None:
    built, nodes, opportunities = _built_prompt()
    transfer = next(o for o in opportunities if o.operator == "transfer")
    response = json.dumps(
        {
            "competitive_set": [
                {
                    "opportunity_id": transfer.opportunity_id,
                    "rank": 1,
                    "reason": "donor mechanism is complementary",
                    "evidence_refs": ["S2-H1", "S2-G", "OP-transfer"],
                    "expected_payoff_horizon": "medium",
                    "semantic_mismatch": None,
                }
            ],
            "not_applicable": [],
        }
    )
    result = parse_critic_response(response, opportunities, built.valid_labels)
    assert result is not None and not result.invalid
    assert result.entries[0].opportunity.opportunity_id == transfer.opportunity_id
    assert result.entries[0].evidence_refs == ("S2-H1", "S2-G", "OP-transfer")


def test_v10_critic_parse_rejects_structural_faults() -> None:
    built, nodes, opportunities = _built_prompt()
    semantic = next(o for o in opportunities if o.operator == "semantic_repair")
    develop = next(o for o in opportunities if o.operator == "develop")
    base = {
        "opportunity_id": develop.opportunity_id,
        "rank": 1,
        "reason": "r",
        "evidence_refs": [],
    }
    assert parse_critic_response(json.dumps({"competitive_set": [base]}), opportunities, built.valid_labels)
    # unknown opportunity id
    assert parse_critic_response(
        json.dumps({"competitive_set": [dict(base, opportunity_id="O99")]}),
        opportunities,
        built.valid_labels,
    ) is None
    # fabricated evidence reference
    assert parse_critic_response(
        json.dumps({"competitive_set": [dict(base, evidence_refs=["S17-H3"])]}),
        opportunities,
        built.valid_labels,
    ) is None
    # semantic repair without a concrete mismatch
    semantic_entry = dict(base, opportunity_id=semantic.opportunity_id)
    assert parse_critic_response(
        json.dumps({"competitive_set": [semantic_entry]}), opportunities, built.valid_labels
    ) is None
    # with a mismatch it passes
    assert parse_critic_response(
        json.dumps({"competitive_set": [dict(semantic_entry, semantic_mismatch="sorted descending but the loop breaks early")]}),
        opportunities,
        built.valid_labels,
    )
    # duplicate ranks
    assert parse_critic_response(
        json.dumps({"competitive_set": [base, dict(base, rank=1)]}), opportunities, built.valid_labels
    ) is None
    # more than K_c entries
    many = [dict(base, rank=rank) for rank in range(1, 6)]
    assert parse_critic_response(
        json.dumps({"competitive_set": many}), opportunities, built.valid_labels
    ) is None
    # empty competitive set
    assert parse_critic_response(
        json.dumps({"competitive_set": []}), opportunities, built.valid_labels
    ) is None
    # not JSON at all
    assert parse_critic_response("no json here", opportunities, built.valid_labels) is None


def test_v10_critic_fallback_picks_highest_quality_develops() -> None:
    nodes = {
        1: _node(1, 10.0, thread=1),
        2: _node(2, 8.0, thread=2),
        3: _node(3, 9.0, thread=3),
    }
    opportunities, _ = build_opportunities(nodes, [1, 2, 3])
    result = fallback_result(nodes, [1, 2, 3], opportunities)
    assert result.invalid
    assert [entry.opportunity.start_id for entry in result.entries] == [1, 3, 2]
    assert [entry.rank for entry in result.entries] == [1, 2, 3]
    assert all(entry.opportunity.operator == "develop" for entry in result.entries)


# ----------------------------------------------------------------------
# Conditioned generation prompts
# ----------------------------------------------------------------------


def test_v10_generation_prompts_follow_information_partition() -> None:
    nodes = {
        1: _node(1, 10.0, thread=1),
        2: _node(2, 9.0, parent=1, thread=1),
        3: _node(3, 8.0, thread=2),
    }
    develop_prompt = build_generation_prompt(
        task_description="Return a high scalar.", operator="develop", nodes=nodes, start_id=2
    )
    assert "code-2" in develop_prompt
    assert "[Formation Path]" in develop_prompt
    assert OPERATOR_INSTRUCTIONS["develop"] in develop_prompt
    assert "code-3" not in develop_prompt  # no donor code for a develop

    transfer_prompt = build_generation_prompt(
        task_description="Return a high scalar.",
        operator="transfer",
        nodes=nodes,
        start_id=2,
        reference_id=3,
    )
    assert "code-2" in transfer_prompt and "code-3" in transfer_prompt
    assert "[Reference Formation Path]" in transfer_prompt

    repair_prompt = build_generation_prompt(
        task_description="Return a high scalar.",
        operator="semantic_repair",
        nodes=nodes,
        start_id=2,
        semantic_mismatch="greedy choice ignores the capacity constraint",
    )
    assert "greedy choice ignores the capacity constraint" in repair_prompt

    restart_prompt = build_generation_prompt(
        task_description="Return a high scalar.",
        operator="restart",
        nodes=nodes,
        start_id=None,
        restart_cards=["Idea: restart card; result: improve; fitness 1 -> 2"],
        template_function=type(
            "Function",
            (),
            {
                "__str__": lambda self: "def f(x):\n    return 0.0\n",
                "__deepcopy__": lambda self, memo: self,
                "body": "",
                "docstring": None,
            },
        )(),
    )
    assert "[Verified Improvements So Far]" in restart_prompt
    assert "restart card" in restart_prompt
    assert "code-2" not in restart_prompt and "code-3" not in restart_prompt
    assert "def f(x):" in restart_prompt  # target signature, not a full program

    for instruction in OPERATOR_INSTRUCTIONS.values():
        assert instruction  # all five operators keep their verbatim instruction


def test_v10_parse_candidate_response_takes_last_code_block() -> None:
    parsed = parse_candidate_response(
        "Idea: some idea\nCode:\n```python\ndef f(x):\n    return 1\n```\ntrailing"
    )
    assert parsed.idea == "some idea"
    assert "return 1" in parsed.code
    fenced = parse_candidate_response('```json\n{"a": 1}\n```\n```python\ndef f(x):\n    return 2\n```')
    assert "return 2" in fenced.code


# ----------------------------------------------------------------------
# Repair accounting
# ----------------------------------------------------------------------


class FailingThenWorkingLLM(LLM):
    """Odd generations return invalid code once, then bounded repairs fix them."""

    def __init__(self) -> None:
        super().__init__()
        self.generation_calls = 0
        self.valid_calls = 0

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        if "valuation critic" in prompt:
            return json.dumps({"competitive_set": []})
        if "Repair the failed realization" in prompt:
            self.valid_calls += 1
            value = 1.0 + 0.1 * self.valid_calls
            return f"Idea: repaired\nCode:\n```python\ndef f(x):\n    return {value}\n```"
        self.generation_calls += 1
        if self.generation_calls % 2 == 1:
            return "Idea: broken\nCode:\n```python\ndef f(x):\n    return [broken\n```"
        self.valid_calls += 1
        value = 1.0 + 0.1 * self.valid_calls
        return f"Idea: fine\nCode:\n```python\ndef f(x):\n    return {value}\n```"


def test_v10_repairs_are_bounded_and_accounted(tmp_path: Path) -> None:
    method = _make_method(tmp_path, FailingThenWorkingLLM(), budget=4, n_roots=2)
    method.run()

    # the critic emits an empty competitive set, so every slot uses the fallback
    assert method._critic_invalid == 2
    assert method._repair_llm_calls >= 1
    assert method._n_calls > method._n_eval  # repair re-evaluations counted separately
    assert method._repair_eval_calls == method._n_calls - method._n_eval
    rows = list(csv.DictReader((tmp_path / "run" / "evaluations.csv").open(newline="")))
    kinds = {row["attempt_kind"] for row in rows}
    assert "repair" in kinds
    assert all(int(row["attempt"]) <= 3 for row in rows)  # at most two repairs


def test_v10_opening_operators_match_spec() -> None:
    assert OPENING_OPERATORS == {"pivot", "transfer", "restart"}
    assert SEMANTIC_REPAIR == "semantic_repair"


# ----------------------------------------------------------------------
# Long-run operations paths
# ----------------------------------------------------------------------


class ExplodingLLM(ToyLLM):
    """Behaves like the toy but dies on the Nth generation call."""

    def __init__(self, fail_at: int) -> None:
        super().__init__()
        self.fail_at = fail_at

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        if "valuation critic" not in prompt:
            if self.calls + 1 == self.fail_at:
                raise RuntimeError("simulated backend outage")
        return super().draw_sample(prompt, *args, **kwargs)


def test_v10_resume_continues_to_budget(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    first = TraceAADV10(
        ExplodingLLM(fail_at=5),
        ToyEvaluation(),
        artifacts=RunArtifacts(run_dir),
        budget=6,
        n_roots=2,
        seed=0,
        checkpoint_dir=run_dir / "checkpoints",
        task_key="toy",
    )
    with pytest.raises(RuntimeError, match="simulated backend outage"):
        first.run()
    crashed_state = json.loads((run_dir / "checkpoints" / "latest.json").read_text(encoding="utf-8"))
    assert 0 < crashed_state["n_eval"] < 6

    resumed = TraceAADV10(
        ToyLLM(),
        ToyEvaluation(),
        artifacts=RunArtifacts(run_dir),
        budget=6,
        n_roots=2,
        seed=0,
        checkpoint_dir=run_dir / "checkpoints",
        resume_from=run_dir / "checkpoints" / "latest.json",
        task_key="toy",
    )
    resumed.run()
    assert resumed._n_eval == 6
    summary = json.loads((run_dir / "logs" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "finished"
    assert summary["budget_slots"] == 6
    rows = list(csv.DictReader((run_dir / "evaluations.csv").open(newline="")))
    slots = [int(row["slot"]) for row in rows]
    assert slots == sorted(slots)
    assert max(slots) == 6


class AlwaysBrokenLLM(LLM):
    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        if "valuation critic" in prompt:
            return json.dumps({"competitive_set": []})
        return "Idea: broken\nCode:\n```python\ndef f(x):\n    return [broken\n```"


def test_v10_initialization_failure_is_explicit(tmp_path: Path) -> None:
    method = _make_method(tmp_path, AlwaysBrokenLLM(), budget=3, n_roots=2)
    method.run()
    summary = json.loads((tmp_path / "run" / "logs" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "initialization_failure"
    assert summary["stop_reason"] == "evaluator_budget_exhausted_during_initialization"
    assert summary["budget_slots"] == 3
    with pytest.raises(RuntimeError, match="not a completed search"):
        from experiments.eval_artifacts import load_run_summary

        load_run_summary(tmp_path / "run")


class CountingLLM(ToyLLM):
    """Toy behaviour plus token counters; the prompt counter fails once."""

    def __init__(self, *, fail_first: bool = False) -> None:
        super().__init__()
        self.fail_first = fail_first
        self.count_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def count_tokens(self, text: str) -> int:
        self.count_calls += 1
        return len(text) // 4

    def count_prompt_tokens(self, prompt: str) -> int:
        if self.fail_first and self.count_calls == 0:
            raise RuntimeError("tokenizer unavailable")
        return len(prompt) // 4


def test_v10_token_accounting_reports_counts(tmp_path: Path) -> None:
    llm = CountingLLM()
    method = _make_method(tmp_path, llm, budget=4, n_roots=2)
    method.run()
    assert method._llm_tokens["generation_prompt"] > 0
    assert method._llm_tokens["critic_prompt"] > 0
    assert method._llm_tokens["generation_completion"] > 0
    summary = json.loads((tmp_path / "run" / "logs" / "summary.json").read_text(encoding="utf-8"))
    assert summary["token_accounting"] == "llm_count_tokens"
    assert summary["critic_prompt_tokens"] > 0


def test_v10_token_counting_failure_does_not_stop_search(tmp_path: Path) -> None:
    llm = CountingLLM(fail_first=True)
    method = _make_method(tmp_path, llm, budget=4, n_roots=2)
    method.run()
    assert method._n_eval == 4
    assert method._token_accounting_failures == 1
    events = [
        json.loads(line)
        for line in (tmp_path / "run" / "mechanism_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(item["event"] == "token_accounting_disabled" for item in events)
    summary = json.loads((tmp_path / "run" / "logs" / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "finished"
    assert summary["token_accounting"] == "disabled_after_failure"


def test_v10_best_history_feeds_the_heldout_loader(tmp_path: Path) -> None:
    from experiments.eval_artifacts import load_scored_samples, pick_best_sample

    method = _make_method(tmp_path, ToyLLM(), budget=5, n_roots=2)
    method.run()
    run_dir = tmp_path / "run"
    records = load_scored_samples(run_dir)
    assert records
    best, all_records = pick_best_sample(run_dir, max_sample_order=3)
    assert best["score"] == max(record["score"] for record in all_records)
    assert all(record["sample_order"] <= 3 for record in all_records)
    assert "def f(x):" in best["program"]


def test_v10_critic_prompt_fits_its_character_budget() -> None:
    big = "code-" + "x = 1\n" * 400  # ~2800 chars per program
    nodes = {}
    threads = {}
    for index in range(1, 13):
        node_id = index
        thread_id = (index - 1) // 2 + 1
        nodes[node_id] = ProgramNode(
            node_id, f"{big}-{index}", float(index), None, thread_id, f"idea {index}", index
        )
        if node_id % 2 == 1:
            threads[thread_id] = Thread(
                thread_id, "init", f"root idea {thread_id}", index, node_id, None, 1, [float(index)]
            )
    attempts: list[AttemptRecord] = []
    opportunities, references = build_opportunities(nodes, [1, 3, 5, 7, 9, 11])
    from llm4ad.method.traceaad_v10.critic import (
        build_critic_prompt as build_prompt,
        critic_char_budget,
    )

    budget = critic_char_budget(32768, 8192)
    built = build_prompt(
        task_description="Return a high scalar.",
        slot=30,
        remaining_budget=970,
        primary_evaluations=29,
        best_node=nodes[11],
        nodes=nodes,
        threads=threads,
        attempts=attempts,
        shortlist=[1, 3, 5, 7, 9, 11],
        opportunities=opportunities,
        references=references,
        char_budget=budget,
    )
    assert len(built.prompt) <= budget
    # A tighter budget still yields a valid prompt with code dropped or clipped.
    tight = build_prompt(
        task_description="Return a high scalar.",
        slot=30,
        remaining_budget=970,
        primary_evaluations=29,
        best_node=nodes[11],
        nodes=nodes,
        threads=threads,
        attempts=attempts,
        shortlist=[1, 3, 5, 7, 9, 11],
        opportunities=opportunities,
        references=references,
        char_budget=25000,
    )
    assert len(tight.prompt) <= 25000
    assert "# [code truncated for the critic]" in tight.prompt or tight.clipped

    cramped = build_prompt(
        task_description="Return a high scalar.",
        slot=30,
        remaining_budget=970,
        primary_evaluations=29,
        best_node=nodes[11],
        nodes=nodes,
        threads=threads,
        attempts=attempts,
        shortlist=[1, 3, 5, 7, 9, 11],
        opportunities=opportunities,
        references=references,
        char_budget=12000,
    )
    assert len(cramped.prompt) <= 12000
    assert "omitted to keep the valuation prompt" in cramped.prompt

    # A budget below the non-code skeleton keeps start code at the floor and
    # drops every reference code without crashing on the None limits.
    starved = build_prompt(
        task_description="Return a high scalar.",
        slot=30,
        remaining_budget=970,
        primary_evaluations=29,
        best_node=nodes[11],
        nodes=nodes,
        threads=threads,
        attempts=attempts,
        shortlist=[1, 3, 5, 7, 9, 11],
        opportunities=opportunities,
        references=references,
        char_budget=2000,
    )
    all_reference_ids = {rid for refs in references.values() for rid in refs}
    assert set(starved.dropped_reference_codes) == all_reference_ids - {1, 3, 5, 7, 9, 11}
    assert "omitted to keep the valuation prompt" in starved.prompt
    assert "# [code truncated for the critic]" in starved.prompt


# ----------------------------------------------------------------------
# Runner integration
# ----------------------------------------------------------------------


def test_v10_launcher_builds_explicit_versioned_plan(tmp_path: Path) -> None:
    from experiments.runners.traceaad.launch_v10 import build_plan, command_for

    plan = build_plan(experiments_root=tmp_path, batch="batch_v10", repeats=1)
    assert len(plan) == 5
    assert plan[0].run_dir == tmp_path / "tsp_construct" / "traceaad_v10" / "batch_v10_tsp_rep1"
    command = command_for(plan[0], "local")
    assert command[command.index("--version") + 1] == "v10"
    assert command[command.index("--n-init") + 1] == "8"
    assert command[command.index("--budget") + 1] == "1000"


def test_v10_runner_spec_and_method_params() -> None:
    from experiments.runners.traceaad.run import _v10_method_params, make_run_spec

    spec = make_run_spec(task="tsp_construct", version="v10", budget=1000)
    assert spec.method_name == "traceaad_v10"
    assert spec.n_init == 8
    assert spec.context_token_limit == 32768
    params = _v10_method_params(spec)
    assert params["mechanism"] == "trajectory_aware_joint_design_opportunity_allocation"
    assert params["operators"] == ["develop", "pivot", "transfer", "restart", "semantic_repair"]
    assert params["K_s"] == 8 and params["K_c"] == 4 and params["H_G"] == [1, 2, 4]
    with pytest.raises(ValueError):
        make_run_spec(task="tsp_construct", version="v10", budget=1000, n_init=4)

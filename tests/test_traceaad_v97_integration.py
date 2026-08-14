from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.method.traceaad_v9_7 import (
    CHECKPOINT_VERSION,
    PROTOCOL_ID,
    TraceAADV97,
)
from llm4ad.method.traceaad_v9_7.checkpoint import (
    _forest_from_dict,
    _forest_to_dict,
)
from llm4ad.method.traceaad_v9_7.forest import SearchForest
from llm4ad.method.traceaad_v9_7.history import (
    drop_oldest_event,
    render_history,
    select_history,
)
from llm4ad.method.traceaad_v9_7.prompt import (
    INTENT_INSTRUCTIONS,
    build_generation_prompt,
)
from llm4ad.method.traceaad_v9_7.schema import (
    AttemptKind,
    AttemptRecord,
    DiffStatistics,
    DirectOutcome,
    GenerationIntent,
    ProgramArtifact,
    REFINE_PROBABILITY,
)
from llm4ad.method.traceaad_v9_7.selection import (
    route_root_id,
    score_routes,
    select_anchor,
)
from llm4ad.method.traceaad_v9_7.traceaad import bootstrap_abs_delta, draw_intent

V96_DIR = Path("llm4ad/method/traceaad_v9_6")
V97_DIR = Path("llm4ad/method/traceaad_v9_7")

EXAMPLE_DIFF = "\n".join(
    [
        "--- parent.py",
        "+++ candidate.py",
        "@@ -1,1 +1,1 @@",
        "-old_line = 1",
        "+new_line = 2",
    ]
)


def _attempt(
    attempt_id: int,
    *,
    anchor_state_id: int,
    candidate_order: int,
    outcome: DirectOutcome,
    idea: str,
    intent: str | None = "refine",
    code_hash: str | None = None,
    child_state_id: int | None = None,
    artifact_id: int | None = None,
) -> AttemptRecord:
    invalid = outcome is DirectOutcome.INVALID
    return AttemptRecord(
        attempt_id=attempt_id,
        status="finalized",
        anchor_state_id=anchor_state_id,
        child_state_id=child_state_id,
        artifact_id=artifact_id,
        intent=intent,
        declared_idea=idea,
        raw_code_hash=code_hash,
        evaluator_input_hash=code_hash,
        actual_diff=None if invalid else EXAMPLE_DIFF,
        diff_statistics=(
            None
            if invalid
            else DiffStatistics(added_lines=1, removed_lines=1, changed_lines=2)
        ),
        parent_fitness=1.0,
        child_fitness=None if invalid else 1.1,
        directed_delta=None,
        direct_outcome=outcome,
        attempt_kind=AttemptKind.INVALID if invalid else AttemptKind.NEW_ARTIFACT,
        failure_category="parse" if invalid else None,
        failure_feedback="missing a fenced Python code block" if invalid else None,
        evaluator_called=not invalid,
        candidate_order=candidate_order,
        creation_time="2026-08-13T00:00:00+00:00",
        stage="search",
        iteration=None,
    )


def _add_route(
    forest: SearchForest,
    *,
    root_fitness: float,
    creation_order: int,
    chain: tuple[float, ...] = (),
) -> list[int]:
    """Add a root plus an optional chain of child states; return state ids."""
    artifact = forest.add_artifact(
        evaluator_input_code=f"root_code_{creation_order}",
        fitness=root_fitness,
        discovery_order=creation_order,
    )
    state = forest.add_root_state(
        artifact_id=artifact.artifact_id, creation_order=creation_order
    )
    state_ids = [state.state_id]
    for offset, fitness in enumerate(chain, start=1):
        child_artifact = forest.add_artifact(
            evaluator_input_code=f"code_{creation_order}_{offset}",
            fitness=fitness,
            discovery_order=creation_order * 100 + offset,
        )
        attempt_id = forest.next_attempt_id()
        child = forest.add_child_state(
            parent_state_id=state.state_id,
            artifact_id=child_artifact.artifact_id,
            attempt_id=attempt_id,
            creation_order=creation_order * 100 + offset,
        )
        forest.add_attempt(
            _attempt(
                attempt_id,
                anchor_state_id=state.state_id,
                candidate_order=creation_order * 100 + offset,
                outcome=DirectOutcome.IMPROVE,
                idea=f"step {offset}",
                code_hash=f"hash_{creation_order}_{offset}",
                child_state_id=child.state_id,
                artifact_id=child_artifact.artifact_id,
            )
        )
        state = child
        state_ids.append(state.state_id)
    return state_ids


# ---------------------------------------------------------------------------
# History: parent improvement path only
# ---------------------------------------------------------------------------


def test_v97_source_is_byte_identical_to_v96() -> None:
    assert (V97_DIR / "source.py").read_bytes() == (V96_DIR / "source.py").read_bytes()


def _add_direct(
    forest: SearchForest,
    anchor_state_id: int,
    *,
    order: int,
    outcome: DirectOutcome,
    idea: str,
    code_hash: str | None,
) -> int:
    attempt_id = forest.next_attempt_id()
    forest.add_attempt(
        _attempt(
            attempt_id,
            anchor_state_id=anchor_state_id,
            candidate_order=order,
            outcome=outcome,
            idea=idea,
            code_hash=code_hash,
        )
    )
    return attempt_id


def test_v97_history_shows_parent_path_and_omits_direct_attempts() -> None:
    forest = SearchForest("contract", maximize=True)
    state_ids = _add_route(forest, root_fitness=1.0, creation_order=1, chain=(1.1, 1.2))
    leaf = state_ids[-1]
    _add_direct(
        forest,
        leaf,
        order=50,
        outcome=DirectOutcome.IMPROVE,
        idea="direct improve",
        code_hash="direct_improve",
    )
    _add_direct(
        forest,
        leaf,
        order=51,
        outcome=DirectOutcome.REGRESS,
        idea="direct regress",
        code_hash="direct_regress",
    )

    selection = select_history(forest, leaf)
    text = render_history(forest, selection)

    assert selection.event_ids == selection.formation_event_ids
    assert len(selection.event_ids) == 2
    assert forest.direct_attempt_ids(leaf)  # facts still exist
    assert "direct improve" not in text
    assert "direct regress" not in text
    assert "Attempt from current algorithm" not in text
    assert text.count("[History ") == 2
    assert "[History 1] Formation step" in text
    assert "[History 2] Formation step" in text
    assert "Idea: step 1" in text
    assert "Idea: step 2" in text


def test_v97_history_keeps_the_most_recent_eight_formation_steps() -> None:
    forest = SearchForest("contract", maximize=True)
    chain = tuple(1.0 + index / 10 for index in range(1, 11))
    state_ids = _add_route(forest, root_fitness=1.0, creation_order=1, chain=chain)
    selection = select_history(forest, state_ids[-1])

    assert len(selection.formation_pool_ids) == 10
    assert len(selection.event_ids) == 8
    assert selection.event_ids == selection.formation_pool_ids[-8:]


def test_v97_history_renders_absence_at_root() -> None:
    forest = SearchForest("contract", maximize=True)
    state_ids = _add_route(forest, root_fitness=1.0, creation_order=1)
    selection = select_history(forest, state_ids[0])

    assert selection.event_ids == ()
    text = render_history(forest, selection)
    assert "No history events are shown for this algorithm." in text


def test_v97_drop_oldest_event_shrinks_parent_path_for_context() -> None:
    forest = SearchForest("contract", maximize=True)
    state_ids = _add_route(forest, root_fitness=1.0, creation_order=1, chain=(1.1, 1.2))
    selection = select_history(forest, state_ids[-1])
    shrunk = drop_oldest_event(selection)

    assert len(selection.event_ids) == 2
    assert shrunk.event_ids == selection.event_ids[1:]
    assert shrunk.formation_event_ids == selection.formation_event_ids[1:]


# ---------------------------------------------------------------------------
# Two-level allocation
# ---------------------------------------------------------------------------


def test_v97_route_scores_sum_generations_and_take_route_best_q() -> None:
    forest = SearchForest("contract", maximize=True)
    route_a = _add_route(forest, root_fitness=1.0, creation_order=1, chain=(2.0, 3.0))
    route_b = _add_route(forest, root_fitness=2.5, creation_order=2)
    forest.get_state(route_a[0]).generation_count_n = 4
    forest.get_state(route_a[1]).generation_count_n = 3
    forest.get_state(route_a[2]).generation_count_n = 2
    forest.get_state(route_b[0]).generation_count_n = 1

    scores = {item.root_state_id: item for item in score_routes(forest, 1.0)}
    a = scores[route_a[0]]
    b = scores[route_b[0]]

    assert a.best_directed_fitness == 3.0
    assert a.generation_count_n == 9
    assert a.score == pytest.approx(3.0 + 1.0 / (10**0.5))
    assert b.best_directed_fitness == 2.5
    assert b.generation_count_n == 1
    assert b.score == pytest.approx(2.5 + 1.0 / (2**0.5))
    for state_id in route_a:
        assert route_root_id(forest, state_id) == route_a[0]


def test_v97_budget_moves_to_less_consumed_route_when_quality_is_close() -> None:
    forest = SearchForest("contract", maximize=True)
    strong = _add_route(forest, root_fitness=100.0, creation_order=1)
    contender = _add_route(forest, root_fitness=98.0, creation_order=2)
    forest.get_state(strong[0]).generation_count_n = 400
    forest.get_state(contender[0]).generation_count_n = 10

    decision = select_anchor(forest, optimism_scale=10.0)

    # strong: 100 + 10/sqrt(401) ~= 100.5; contender: 98 + 10/sqrt(11) ~= 101.0
    assert decision.root_state_id == contender[0]
    assert decision.state_id == contender[0]


def test_v97_anchor_argmax_is_restricted_to_the_selected_route() -> None:
    forest = SearchForest("contract", maximize=True)
    # Route A holds the globally best anchor but is heavily consumed.
    route_a = _add_route(forest, root_fitness=100.0, creation_order=1, chain=(120.0,))
    route_b = _add_route(forest, root_fitness=119.0, creation_order=2)
    forest.get_state(route_a[0]).generation_count_n = 200
    forest.get_state(route_a[1]).generation_count_n = 200
    forest.get_state(route_b[0]).generation_count_n = 0

    decision = select_anchor(forest, optimism_scale=50.0)

    # Route B wins on route score; the globally best anchor (120, in route A)
    # must not leak into the anchor-level argmax.
    assert decision.root_state_id == route_b[0]
    assert decision.state_id == route_b[0]
    assert all(
        route_root_id(forest, item.state_id) == route_b[0]
        for item in decision.state_scores
    )


def test_v97_route_tie_breaks_prefer_less_consumed_then_earlier_root() -> None:
    forest = SearchForest("contract", maximize=True)
    first = _add_route(forest, root_fitness=1.0, creation_order=1)
    second = _add_route(forest, root_fitness=1.0, creation_order=2)

    # Identical q and N: the earlier-created root wins.
    decision = select_anchor(forest, optimism_scale=0.0)
    assert decision.root_state_id == first[0]

    # Same score, more consumption on the first: the less consumed route wins.
    forest.get_state(first[0]).generation_count_n = 5
    decision = select_anchor(forest, optimism_scale=0.0)
    assert decision.root_state_id == second[0]


def test_v97_single_route_reduces_to_v96_anchor_rule() -> None:
    forest = SearchForest("contract", maximize=True)
    state_ids = _add_route(forest, root_fitness=1.0, creation_order=1, chain=(1.5,))
    forest.get_state(state_ids[0]).generation_count_n = 3

    decision = select_anchor(forest, optimism_scale=0.5)

    # child: 1.5 + 0.5/1 = 2.0; root: 1.0 + 0.5/2 = 1.25
    assert decision.state_id == state_ids[1]


def test_v97_bootstrap_scale_includes_valid_zero_delta_child() -> None:
    assert bootstrap_abs_delta(child_created=True, directed_delta=0.0) == 0.0
    assert bootstrap_abs_delta(child_created=True, directed_delta=-1.25) == 1.25
    assert bootstrap_abs_delta(child_created=False, directed_delta=0.0) is None


# ---------------------------------------------------------------------------
# Generation intents
# ---------------------------------------------------------------------------


def test_v97_intent_draw_is_deterministic_and_close_to_fixed_mixture() -> None:
    draws = [draw_intent(0, iteration) for iteration in range(2000)]
    again = [draw_intent(0, iteration) for iteration in range(2000)]
    assert draws == again
    assert set(draws) == {GenerationIntent.REFINE, GenerationIntent.EXPLORE}
    refine_rate = draws.count(GenerationIntent.REFINE) / len(draws)
    assert abs(refine_rate - REFINE_PROBABILITY) < 0.03

    other_seed = [draw_intent(1, iteration) for iteration in range(2000)]
    assert other_seed != draws


def test_v97_intents_share_context_and_differ_only_in_instruction() -> None:
    anchor = ProgramArtifact(
        artifact_id=0,
        evaluator_contract_hash="contract",
        evaluator_input_hash="hash",
        evaluator_input_code="def f():\n    return 1",
        fitness=1.25,
        directed_fitness=1.25,
        code_length=20,
        program_loc=2,
        first_discovery_order=1,
    )
    history_text = "[Recent Algorithm Improvement History]\n\n[History 1] Formation step\nIdea: x\nChange: +1/-1 lines\nResult: improve\nFitness: 1 -> 1.25"
    prompts = {
        intent: build_generation_prompt(
            task_description="Solve the task.",
            anchor=anchor,
            history_text=history_text,
            intent=intent,
            maximize=True,
        )
        for intent in GenerationIntent
    }

    refine = prompts[GenerationIntent.REFINE]
    explore = prompts[GenerationIntent.EXPLORE]
    assert INTENT_INSTRUCTIONS[GenerationIntent.REFINE] in refine
    assert INTENT_INSTRUCTIONS[GenerationIntent.EXPLORE] in explore
    assert "within its existing design" in refine
    assert "materially different way" in explore

    # Identical prefix: task, current algorithm, and history are shared facts.
    refine_prefix = refine.split("[Instruction]")[0]
    explore_prefix = explore.split("[Instruction]")[0]
    assert refine_prefix == explore_prefix
    assert history_text in refine_prefix

    # Identical suffix after the intent sentence (contract lines).
    refine_tail = refine.split("[Instruction]")[1].split("\n", 2)[2]
    explore_tail = explore.split("[Instruction]")[1].split("\n", 2)[2]
    assert refine_tail == explore_tail


# ---------------------------------------------------------------------------
# Checkpoint roundtrip with intent
# ---------------------------------------------------------------------------


def test_v97_forest_checkpoint_roundtrip_preserves_intent() -> None:
    forest = SearchForest("contract", maximize=True)
    state_ids = _add_route(forest, root_fitness=1.0, creation_order=1, chain=(1.1,))
    explore_attempt = forest.next_attempt_id()
    forest.add_attempt(
        _attempt(
            explore_attempt,
            anchor_state_id=state_ids[-1],
            candidate_order=999,
            outcome=DirectOutcome.REGRESS,
            idea="big restructure",
            intent="explore",
            code_hash="explore_hash",
        )
    )

    restored = _forest_from_dict(json.loads(json.dumps(_forest_to_dict(forest))))

    assert restored.get_attempt(explore_attempt).intent == "explore"
    assert {a.intent for a in restored.attempts()} == {"refine", "explore"}
    assert _forest_to_dict(restored) == _forest_to_dict(forest)


# ---------------------------------------------------------------------------
# Runner integration
# ---------------------------------------------------------------------------


def test_v97_runner_builds_complete_frozen_method(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_7",
        budget=1000,
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")

    assert isinstance(method, TraceAADV97)
    assert spec.method_name == "traceaad_v9_7"
    assert spec.n_init == 8
    assert spec.context_token_limit == 32768
    assert spec.llm_output_tokens == 8192
    assert method.search_configuration() == run._v97_method_params(spec)
    assert method.search_configuration()["protocol_id"] == PROTOCOL_ID
    assert method.search_configuration()["checkpoint_schema_version"] == (
        CHECKPOINT_VERSION
    )
    assert method.search_configuration()["budget_unit"] == "real_evaluator_call"
    assert method.search_configuration()["refine_probability"] == 0.7
    assert method.search_configuration()["explore_probability"] == pytest.approx(0.3)
    assert method.search_configuration()["history_selector_id"] == (
        "v97_recent_parent_formation_path_v1"
    )

    # Budget counts real evaluator calls, not completed responses.
    method._candidate_count = 5000
    assert method._has_budget()
    method._evaluation_count = 1000
    assert not method._has_budget()
    method._llm.close()


def test_v97_run_config_records_logical_generator_without_service_source(
    tmp_path: Path,
) -> None:
    spec = run.make_run_spec(
        task="online_bin_packing",
        version="v9_7",
        backend="server3",
        budget=1000,
        repeat=2,
        run_name="v9_7_obp_rep2",
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert not resumed
    assert payload["method"] == "traceaad_v9_7"
    assert payload["method_params"] == run._v97_method_params(spec)
    assert payload["method_params"]["budget_unit"] == "real_evaluator_call"
    assert payload["method_params"]["intent_policy_id"].startswith("v97_fixed_mixture")
    assert payload["generator_environment"]["logical_model_name"] == "Qwen3.6-27B"
    assert payload["generator_environment"]["max_total_context"] == 32768
    assert payload["generator_environment"]["max_new_tokens"] == 8192
    assert "backend" not in payload
    assert "llm" not in payload
    assert "base_url" not in json.dumps(payload)
    assert "quant" not in json.dumps(payload).lower()


def test_v97_resume_accepts_only_matching_protocol(tmp_path: Path) -> None:
    original = run.make_run_spec(
        task="tsp_construct",
        version="v9_7",
        budget=1000,
        run_name="matching_v97",
        experiments_root=tmp_path,
    )
    run_dir, run_name, _ = run.resolve_run_dir(original)
    run.write_run_config(original, run_dir, run_name)
    resumed_spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_7",
        budget=1000,
        resume_from=run_dir,
        experiments_root=tmp_path,
    )

    resolved, _, resumed = run.resolve_run_dir(resumed_spec)

    assert resumed
    assert resolved == run_dir


def test_v97_official_runner_fixes_root_count_to_eight(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly eight"):
        run.make_run_spec(
            task="tsp_construct",
            version="v9_7",
            n_init=10,
            experiments_root=tmp_path,
        )

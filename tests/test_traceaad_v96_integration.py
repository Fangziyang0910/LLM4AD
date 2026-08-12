from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.method.traceaad_v9_6 import (
    CHECKPOINT_VERSION,
    PROTOCOL_ID,
    TraceAADV96,
)
from llm4ad.method.traceaad_v9_6.forest import SearchForest
from llm4ad.method.traceaad_v9_6.history import (
    drop_oldest_event,
    render_history,
    select_history,
)
from llm4ad.method.traceaad_v9_6.schema import (
    AttemptKind,
    AttemptRecord,
    DiffStatistics,
    DirectOutcome,
)

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
    code_hash: str | None = None,
    child_state_id: int | None = None,
    artifact_id: int | None = None,
    parent_fitness: float | None = 1.0,
    child_fitness: float | None = 1.1,
) -> AttemptRecord:
    invalid = outcome is DirectOutcome.INVALID
    return AttemptRecord(
        attempt_id=attempt_id,
        status="finalized",
        anchor_state_id=anchor_state_id,
        child_state_id=child_state_id,
        artifact_id=artifact_id,
        declared_idea=idea,
        raw_code_hash=code_hash,
        evaluator_input_hash=code_hash,
        actual_diff=None if invalid else EXAMPLE_DIFF,
        diff_statistics=(
            None
            if invalid
            else DiffStatistics(added_lines=1, removed_lines=1, changed_lines=2)
        ),
        parent_fitness=parent_fitness,
        child_fitness=None if invalid else child_fitness,
        directed_delta=None,
        direct_outcome=outcome,
        attempt_kind=AttemptKind.INVALID if invalid else AttemptKind.NEW_ARTIFACT,
        failure_category="parse" if invalid else None,
        failure_feedback="missing a fenced Python code block" if invalid else None,
        evaluator_called=not invalid,
        candidate_order=candidate_order,
        creation_time="2026-08-12T00:00:00+00:00",
        stage="search",
        iteration=None,
    )


def _forest_with_formation_chain(steps: int) -> tuple[SearchForest, int, list[int]]:
    """Build root -> chain of `steps` formation states; return anchor and ids."""
    forest = SearchForest("contract", maximize=True)
    artifact = forest.add_artifact(
        evaluator_input_code="code_root", fitness=1.0, discovery_order=1
    )
    state = forest.add_root_state(artifact_id=artifact.artifact_id, creation_order=1)
    formation_ids: list[int] = []
    for index in range(1, steps + 1):
        child_artifact = forest.add_artifact(
            evaluator_input_code=f"code_{index}",
            fitness=1.0 + index / 10,
            discovery_order=1 + index,
        )
        attempt_id = forest.next_attempt_id()
        child_state = forest.add_child_state(
            parent_state_id=state.state_id,
            artifact_id=child_artifact.artifact_id,
            attempt_id=attempt_id,
            creation_order=1 + index,
        )
        forest.add_attempt(
            _attempt(
                attempt_id,
                anchor_state_id=state.state_id,
                candidate_order=1 + index,
                outcome=DirectOutcome.IMPROVE,
                idea=f"formation {index}",
                code_hash=f"formation_hash_{index}",
                child_state_id=child_state.state_id,
                artifact_id=child_artifact.artifact_id,
            )
        )
        formation_ids.append(attempt_id)
        state = child_state
    return forest, state.state_id, formation_ids


def _add_direct(
    forest: SearchForest,
    anchor_state_id: int,
    *,
    order: int,
    outcome: DirectOutcome,
    code_hash: str | None,
    idea: str,
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


def test_v96_selection_caps_direct_and_fills_with_recent_formation() -> None:
    forest, anchor_id, formation_ids = _forest_with_formation_chain(3)
    d1 = _add_direct(
        forest, anchor_id, order=10, outcome=DirectOutcome.IMPROVE,
        code_hash="X", idea="improve one",
    )
    d2 = _add_direct(
        forest, anchor_id, order=11, outcome=DirectOutcome.IMPROVE,
        code_hash="X", idea="improve one again",
    )
    d3 = _add_direct(
        forest, anchor_id, order=12, outcome=DirectOutcome.IMPROVE,
        code_hash="Y", idea="improve two",
    )
    _add_direct(
        forest, anchor_id, order=13, outcome=DirectOutcome.REGRESS,
        code_hash="Z", idea="regress old",
    )
    _add_direct(
        forest, anchor_id, order=14, outcome=DirectOutcome.PLATEAU,
        code_hash="P", idea="plateau",
    )
    _add_direct(
        forest, anchor_id, order=15, outcome=DirectOutcome.INVALID,
        code_hash=None, idea="invalid",
    )
    d7 = _add_direct(
        forest, anchor_id, order=16, outcome=DirectOutcome.REGRESS,
        code_hash="W", idea="regress recent one",
    )
    d8 = _add_direct(
        forest, anchor_id, order=17, outcome=DirectOutcome.REGRESS,
        code_hash="V", idea="regress recent two",
    )

    selection = select_history(forest, anchor_id)

    # Identical code (d1/d2) counts once; caps are 2 improved + 2 regressed.
    assert selection.direct_event_ids == (d2, d3, d7, d8)
    assert d1 not in selection.event_ids
    assert selection.formation_event_ids == tuple(formation_ids)
    assert selection.event_ids == (*formation_ids, d2, d3, d7, d8)
    assert len(selection.direct_pool_ids) == 8

    text = render_history(forest, selection)
    assert text.count("[History ") == 7
    assert "[History 1] Formation step" in text
    assert "[History 4] Attempt from current algorithm" in text
    assert "Change: +1/-1 lines; removed: `old_line = 1`; added: `new_line = 2`" in text
    assert "Result: regress" in text
    assert "Fitness: 1 -> 1.1" in text
    assert "invalid" not in text  # plateau/invalid attempts never render
    assert "plateau" not in text


def test_v96_selection_uses_recent_formation_when_no_direct_history() -> None:
    forest, anchor_id, formation_ids = _forest_with_formation_chain(10)

    selection = select_history(forest, anchor_id)

    assert selection.direct_event_ids == ()
    assert selection.event_ids == tuple(formation_ids[-8:])

    text = render_history(forest, selection)
    assert text.count("[History ") == 8
    assert "Attempt from current algorithm" not in text


def test_v96_render_states_absence_of_history() -> None:
    forest = SearchForest("contract", maximize=True)
    artifact = forest.add_artifact(
        evaluator_input_code="code_root", fitness=1.0, discovery_order=1
    )
    root = forest.add_root_state(artifact_id=artifact.artifact_id, creation_order=1)

    selection = select_history(forest, root.state_id)

    assert selection.event_ids == ()
    text = render_history(forest, selection)
    assert "No history events are shown for this algorithm." in text


def test_v96_drop_oldest_event_shrinks_selection_for_context() -> None:
    forest, anchor_id, formation_ids = _forest_with_formation_chain(3)

    selection = select_history(forest, anchor_id)
    shrunk = drop_oldest_event(selection)

    assert shrunk.event_ids == selection.event_ids[1:]
    assert formation_ids[0] not in shrunk.formation_event_ids
    assert shrunk.formation_pool_ids == selection.formation_pool_ids
    text = render_history(forest, shrunk)
    assert text.count("[History ") == 2


def test_v96_runner_builds_complete_frozen_method(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_6",
        budget=1000,
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")

    assert isinstance(method, TraceAADV96)
    assert spec.method_name == "traceaad_v9_6"
    assert spec.n_init == 8
    assert spec.context_token_limit == 32768
    assert spec.llm_output_tokens == 8192
    assert method.search_configuration() == run._v96_method_params(spec)
    assert method.search_configuration()["protocol_id"] == PROTOCOL_ID
    assert method.search_configuration()["checkpoint_schema_version"] == (
        CHECKPOINT_VERSION
    )
    assert method.search_configuration()["budget_unit"] == "real_evaluator_call"
    assert "diff_excerpt_chars" not in method.search_configuration()

    # Budget counts real evaluator calls, not completed responses.
    method._candidate_count = 5000
    assert method._has_budget()
    method._evaluation_count = 1000
    assert not method._has_budget()
    method._llm.close()


def test_v96_run_config_records_logical_generator_without_service_source(
    tmp_path: Path,
) -> None:
    spec = run.make_run_spec(
        task="online_bin_packing",
        version="v9_6",
        backend="server3",
        budget=1000,
        repeat=2,
        run_name="v9_6_obp_rep2",
        experiments_root=tmp_path,
    )
    run_dir, run_name, resumed = run.resolve_run_dir(spec)
    run.write_run_config(spec, run_dir, run_name)
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))

    assert not resumed
    assert payload["method"] == "traceaad_v9_6"
    assert payload["method_params"] == run._v96_method_params(spec)
    assert payload["method_params"]["budget_unit"] == "real_evaluator_call"
    assert payload["generator_environment"]["logical_model_name"] == "Qwen3.6-27B"
    assert payload["generator_environment"]["max_total_context"] == 32768
    assert payload["generator_environment"]["max_new_tokens"] == 8192
    assert "backend" not in payload
    assert "llm" not in payload
    assert "base_url" not in json.dumps(payload)
    assert "quant" not in json.dumps(payload).lower()


def test_v96_resume_accepts_only_matching_protocol(tmp_path: Path) -> None:
    original = run.make_run_spec(
        task="tsp_construct",
        version="v9_6",
        budget=1000,
        run_name="matching_v96",
        experiments_root=tmp_path,
    )
    run_dir, run_name, _ = run.resolve_run_dir(original)
    run.write_run_config(original, run_dir, run_name)
    resumed_spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_6",
        budget=1000,
        resume_from=run_dir,
        experiments_root=tmp_path,
    )

    resolved, _, resumed = run.resolve_run_dir(resumed_spec)

    assert resumed
    assert resolved == run_dir


def test_v96_official_runner_fixes_root_count_to_eight(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly eight"):
        run.make_run_spec(
            task="tsp_construct",
            version="v9_6",
            n_init=10,
            experiments_root=tmp_path,
        )

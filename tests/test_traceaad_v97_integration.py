from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.method.traceaad_v9_7 import TraceAADV97
from llm4ad.method.traceaad_v9_7.forest import Forest
from llm4ad.method.traceaad_v9_7.history import parent_path, render_path
from llm4ad.method.traceaad_v9_7.prompt import INTENT_INSTRUCTIONS, build_generation_prompt
from llm4ad.method.traceaad_v9_7.schema import Attempt, Intent, Outcome, REFINE_PROBABILITY
from llm4ad.method.traceaad_v9_7.selection import score_routes, select
from llm4ad.method.traceaad_v9_7.traceaad import draw_intent

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
    anchor_id: int,
    order: int,
    outcome: Outcome,
    idea: str,
    intent: str | None = "refine",
    child_id: int | None = None,
    program_id: int | None = None,
) -> Attempt:
    invalid = outcome is Outcome.INVALID
    return Attempt(
        id=attempt_id,
        anchor_id=anchor_id,
        child_id=child_id,
        program_id=program_id,
        intent=intent,
        idea=idea,
        diff=None if invalid else EXAMPLE_DIFF,
        added=0 if invalid else 1,
        removed=0 if invalid else 1,
        parent_fitness=1.0,
        child_fitness=None if invalid else 1.1,
        dq=None,
        outcome=outcome,
        kind="invalid" if invalid else "new",
        order=order,
        stage="search",
        iteration=None,
    )


def _add_route(
    forest: Forest,
    *,
    root_fitness: float,
    order: int,
    chain: tuple[float, ...] = (),
) -> list[int]:
    """Add a root plus an optional chain of child anchors; return anchor ids."""
    program = forest.add_program(
        code=f"root_code_{order}",
        fitness=root_fitness,
        order=order,
    )
    anchor = forest.add_root(program_id=program.id, order=order)
    ids = [anchor.id]
    for offset, fitness in enumerate(chain, start=1):
        child_program = forest.add_program(
            code=f"code_{order}_{offset}",
            fitness=fitness,
            order=order * 100 + offset,
        )
        attempt_id = forest.next_attempt_id()
        child = forest.add_child(
            parent_id=anchor.id,
            program_id=child_program.id,
            attempt_id=attempt_id,
            order=order * 100 + offset,
        )
        forest.add_attempt(
            _attempt(
                attempt_id,
                anchor_id=anchor.id,
                order=order * 100 + offset,
                outcome=Outcome.IMPROVE,
                idea=f"step {offset}",
                child_id=child.id,
                program_id=child_program.id,
            )
        )
        anchor = child
        ids.append(anchor.id)
    return ids


def _add_direct(
    forest: Forest,
    anchor_id: int,
    *,
    order: int,
    outcome: Outcome,
    idea: str,
) -> int:
    attempt_id = forest.next_attempt_id()
    forest.add_attempt(
        _attempt(
            attempt_id,
            anchor_id=anchor_id,
            order=order,
            outcome=outcome,
            idea=idea,
        )
    )
    return attempt_id


def test_v97_history_shows_parent_path_and_omits_direct_attempts() -> None:
    forest = Forest(maximize=True)
    ids = _add_route(forest, root_fitness=1.0, order=1, chain=(1.1, 1.2))
    leaf = ids[-1]
    _add_direct(forest, leaf, order=50, outcome=Outcome.IMPROVE, idea="direct improve")
    _add_direct(forest, leaf, order=51, outcome=Outcome.REGRESS, idea="direct regress")

    shown = parent_path(forest, leaf)
    text = render_path(forest, shown)

    assert len(shown) == 2
    assert "direct improve" not in text
    assert "direct regress" not in text
    assert "Attempt from current algorithm" not in text
    assert text.count("[History ") == 2
    assert "[History 1] Formation step" in text
    assert "[History 2] Formation step" in text
    assert "Idea: step 1" in text
    assert "Idea: step 2" in text


def test_v97_history_keeps_the_most_recent_eight_formation_steps() -> None:
    forest = Forest(maximize=True)
    chain = tuple(1.0 + index / 10 for index in range(1, 11))
    ids = _add_route(forest, root_fitness=1.0, order=1, chain=chain)
    shown = parent_path(forest, ids[-1])

    assert len(forest.parent_path_ids(ids[-1])) == 10
    assert len(shown) == 8
    assert shown == forest.parent_path_ids(ids[-1])[-8:]


def test_v97_history_renders_absence_at_root() -> None:
    forest = Forest(maximize=True)
    ids = _add_route(forest, root_fitness=1.0, order=1)
    shown = parent_path(forest, ids[0])

    assert shown == ()
    text = render_path(forest, shown)
    assert "No history events are shown for this algorithm." in text


def test_v97_route_scores_sum_generations_and_take_route_best_q() -> None:
    forest = Forest(maximize=True)
    route_a = _add_route(forest, root_fitness=1.0, order=1, chain=(2.0, 3.0))
    route_b = _add_route(forest, root_fitness=2.5, order=2)
    forest.get_anchor(route_a[0]).n = 4
    forest.get_anchor(route_a[1]).n = 3
    forest.get_anchor(route_a[2]).n = 2
    forest.get_anchor(route_b[0]).n = 1

    scores = {item.id: item for item in score_routes(forest, 1.0)}
    a = scores[route_a[0]]
    b = scores[route_b[0]]

    assert a.q == 3.0
    assert a.n == 9
    assert a.score == pytest.approx(3.0 + 1.0 / (10**0.5))
    assert b.q == 2.5
    assert b.n == 1
    assert b.score == pytest.approx(2.5 + 1.0 / (2**0.5))
    for anchor_id in route_a:
        assert forest.get_anchor(anchor_id).root_id == route_a[0]


def test_v97_budget_moves_to_less_consumed_route_when_quality_is_close() -> None:
    forest = Forest(maximize=True)
    strong = _add_route(forest, root_fitness=100.0, order=1)
    contender = _add_route(forest, root_fitness=98.0, order=2)
    forest.get_anchor(strong[0]).n = 400
    forest.get_anchor(contender[0]).n = 10

    choice = select(forest, 10.0)

    # strong: 100 + 10/sqrt(401) ~= 100.5; contender: 98 + 10/sqrt(11) ~= 101.0
    assert choice.route_id == contender[0]
    assert choice.anchor_id == contender[0]


def test_v97_anchor_argmax_is_restricted_to_the_selected_route() -> None:
    forest = Forest(maximize=True)
    route_a = _add_route(forest, root_fitness=100.0, order=1, chain=(120.0,))
    route_b = _add_route(forest, root_fitness=119.0, order=2)
    forest.get_anchor(route_a[0]).n = 200
    forest.get_anchor(route_a[1]).n = 200
    forest.get_anchor(route_b[0]).n = 0

    choice = select(forest, 50.0)

    assert choice.route_id == route_b[0]
    assert choice.anchor_id == route_b[0]
    assert all(
        forest.get_anchor(item.id).root_id == route_b[0] for item in choice.anchors
    )


def test_v97_route_tie_breaks_prefer_less_consumed_then_earlier_root() -> None:
    forest = Forest(maximize=True)
    first = _add_route(forest, root_fitness=1.0, order=1)
    second = _add_route(forest, root_fitness=1.0, order=2)

    choice = select(forest, 0.0)
    assert choice.route_id == first[0]

    forest.get_anchor(first[0]).n = 5
    choice = select(forest, 0.0)
    assert choice.route_id == second[0]


def test_v97_single_route_reduces_to_v96_anchor_rule() -> None:
    forest = Forest(maximize=True)
    ids = _add_route(forest, root_fitness=1.0, order=1, chain=(1.5,))
    forest.get_anchor(ids[0]).n = 3

    choice = select(forest, 0.5)

    # child: 1.5 + 0.5/1 = 2.0; root: 1.0 + 0.5/2 = 1.25
    assert choice.anchor_id == ids[1]


def test_v97_intent_draw_is_deterministic_and_close_to_fixed_mixture() -> None:
    draws = [draw_intent(0, iteration) for iteration in range(2000)]
    again = [draw_intent(0, iteration) for iteration in range(2000)]
    assert draws == again
    assert set(draws) == {Intent.REFINE, Intent.EXPLORE}
    refine_rate = draws.count(Intent.REFINE) / len(draws)
    assert abs(refine_rate - REFINE_PROBABILITY) < 0.03

    other_seed = [draw_intent(1, iteration) for iteration in range(2000)]
    assert other_seed != draws


def test_v97_intents_share_context_and_differ_only_in_instruction() -> None:
    history_text = (
        "[Recent Algorithm Improvement History]\n\n"
        "[History 1] Formation step\nIdea: x\nChange: +1/-1 lines\n"
        "Result: improve\nFitness: 1 -> 1.25"
    )
    prompts = {
        intent: build_generation_prompt(
            task_description="Solve the task.",
            code="def f():\n    return 1",
            fitness=1.25,
            history_text=history_text,
            intent=intent,
            maximize=True,
        )
        for intent in Intent
    }

    refine = prompts[Intent.REFINE]
    explore = prompts[Intent.EXPLORE]
    assert INTENT_INSTRUCTIONS[Intent.REFINE] in refine
    assert INTENT_INSTRUCTIONS[Intent.EXPLORE] in explore
    assert "within its existing design" in refine
    assert "materially different way" in explore

    refine_prefix = refine.split("[Instruction]")[0]
    explore_prefix = explore.split("[Instruction]")[0]
    assert refine_prefix == explore_prefix
    assert history_text in refine_prefix

    refine_tail = refine.split("[Instruction]")[1].split("\n", 2)[2]
    explore_tail = explore.split("[Instruction]")[1].split("\n", 2)[2]
    assert refine_tail == explore_tail


def test_v97_forest_checkpoint_roundtrip_preserves_intent() -> None:
    forest = Forest(maximize=True)
    ids = _add_route(forest, root_fitness=1.0, order=1, chain=(1.1,))
    explore_attempt = forest.next_attempt_id()
    forest.add_attempt(
        _attempt(
            explore_attempt,
            anchor_id=ids[-1],
            order=999,
            outcome=Outcome.REGRESS,
            idea="big restructure",
            intent="explore",
        )
    )

    restored = Forest.from_dict(json.loads(json.dumps(forest.to_dict())))

    assert restored.get_attempt(explore_attempt).intent == "explore"
    assert {a.intent for a in restored.attempts()} == {"refine", "explore"}
    assert restored.to_dict() == forest.to_dict()


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
    params = run._v97_method_params(spec)
    assert params["refine_probability"] == 0.7
    assert params["explore_probability"] == pytest.approx(0.3)
    assert "context_limit" not in params

    method._n_candidates = 5000
    assert method._has_budget()
    method._n_eval = 1000
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
    assert payload["method_params"]["refine_probability"] == 0.7
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

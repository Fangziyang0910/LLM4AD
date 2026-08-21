from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.analysis.analyze_v97_search_geometry import (
    macro_family as analysis_macro_family,
    mechanism_tags as analysis_mechanism_tags,
)
from experiments.runners.traceaad import run
from experiments.runners.traceaad.launch_v913 import (
    PREFIX_BUDGET,
    _unit_for_prefix,
    audit_fork,
    fork_prefix,
)
from experiments.runners.traceaad.v913_stage_p import (
    CONDITIONS,
    EVAL_INTERVALS,
    _interval_of,
    _next_selection,
    build_schedule,
    build_trial_prompt,
    trim_forest,
)
from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_7.prompt import (
    build_generation_prompt as build_v97_prompt,
)
from llm4ad.method.traceaad_v9_7.traceaad import draw_intent as draw_intent_v97
from llm4ad.method.traceaad_v9_13 import (
    FRONTIER_ACTIVATION_EVALS,
    Treatment,
)
from llm4ad.method.traceaad_v9_13.checkpoint import load_checkpoint, save_checkpoint
from llm4ad.method.traceaad_v9_13.forest import Forest
from llm4ad.method.traceaad_v9_13.prompt import build_generation_prompt
from llm4ad.method.traceaad_v9_13.regions import RegionView
from llm4ad.method.traceaad_v9_13.schema import Attempt, Intent, Outcome, Program
from llm4ad.method.traceaad_v9_13.traceaad import TraceAADV913, draw_intent

TEMPLATE = """def choose(value: int) -> int:
    return value
"""

HISTORY_TEXT = (
    "[Recent Algorithm Improvement History]\n\n"
    "[History 1] Formation step\nIdea: x\nChange: +1/-1 lines\n"
    "Result: improve\nFitness: 1 -> 1.25"
)


class ScriptedLLM(LLM):
    def __init__(self, start: int = 0) -> None:
        super().__init__()
        self.calls = start

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        return (
            f"Idea: candidate {self.calls}\nCode:\n```python\ndef choose(value: int) -> int:\n"
            f"    return value + {self.calls}\n```"
        )

    def count_tokens(self, text: str) -> int:
        return len(text) // 4 + 1


class ConstantEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            safe_evaluate=False,
            timeout_seconds=10,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return 1.0


def _method(**overrides) -> TraceAADV913:
    kwargs = dict(
        llm=ScriptedLLM(),
        evaluation=ConstantEvaluation(),
        budget=10,
        task_key="tsp_construct",
        n_roots=1,
        seed=0,
    )
    kwargs.update(overrides)
    return TraceAADV913(**kwargs)


def _program(pid: int, code: str, fitness: float, order: int) -> Program:
    return Program(id=pid, code=code, fitness=fitness, q=fitness, length=len(code), order=order)


def _seeded_forest_method(treatment: str = "fp") -> TraceAADV913:
    """A method with one rooted anchor, initialization complete, s=1."""

    method = _method(treatment=treatment, budget=1000)
    program = method._forest.add_program(code=TEMPLATE, fitness=1.0, order=1)
    root = method._forest.add_root(program_id=program.id, order=1)
    method._s = 1.0
    method._initialization_complete = True
    method._bootstrapped = {root.id}
    return method


def test_v913_intent_schedule_is_the_v97_schedule() -> None:
    for seed in (None, 0, 1, 42):
        assert [draw_intent(seed, i) for i in range(400)] == [
            draw_intent_v97(seed, i) for i in range(400)
        ]


def test_v913_prompt_without_global_context_is_byte_identical_to_v97() -> None:
    shared = dict(
        task_description="Solve the task.",
        code="def f():\n    return 1",
        fitness=1.25,
        history_text=HISTORY_TEXT,
        maximize=True,
    )
    for intent in Intent:
        assert build_generation_prompt(intent=intent, **shared) == build_v97_prompt(
            intent=intent, **shared
        )


def test_v913_global_context_sits_between_history_and_instruction() -> None:
    prompt = build_generation_prompt(
        task_description="Solve the task.",
        code="def f():\n    return 1",
        fitness=1.25,
        history_text=HISTORY_TEXT,
        intent=Intent.EXPLORE,
        maximize=True,
        global_context="[Searched Proxy Regions]\nrow",
    )
    assert prompt.index(HISTORY_TEXT) < prompt.index("[Searched Proxy Regions]")
    assert prompt.index("[Searched Proxy Regions]") < prompt.index("[Instruction]")


def test_v913_region_view_frontier_and_tie_breaks() -> None:
    view = RegionView("tsp_construct")
    long_two_opt = (
        "def f():\n    x = 1\n" * 5 + "\ndef g():\n    return apply_2_opt(d)"
    )
    view.record(_program(1, long_two_opt, -6.0, 1))
    view.record(_program(2, "def f():\n    return apply_2_opt(d)", -6.0, 2))
    view.record(_program(3, "def b():\n    return beam_width(x)", -5.5, 3))

    rows = view.frontier_rows()
    assert [row.family for row in rows] == ["explicit_search", "completion_rollout"]
    completion = rows[1]
    # same q, same family: shorter code wins regardless of later evaluation
    assert completion.program_id == 2


def test_v913_frontier_table_is_floor_shaped() -> None:
    view = RegionView("tsp_construct")
    view.record(_program(1, "def f():\n    return apply_2_opt(d)", -6.0, 1))
    view.record(_program(2, "def b():\n    return beam_width(x)", -5.5, 2))
    text = view.frontier_text("completion_rollout")
    assert text.startswith("[Searched Proxy Regions]")
    assert "merely rebuilds a region below its recorded level wastes budget" in text
    # own region first, with frontier tags and quality
    assert "[Current Algorithm's Region]" in text
    assert "Observed tags of frontier program: two_opt" in text
    assert "Directed quality: -6" in text
    # other region: frontier tags + quality, floor semantics, no code/counts
    assert "[Other Searched Regions]" in text
    assert "Region 2 observed tags of frontier program: beam_search" in text
    assert "Region 2 directed quality: -5.5" in text
    assert "```python" not in text and "programs tried" not in text
    assert "Global best directed quality across all regions: -5.5" in text
    # own region still leads the table regardless of quality order
    other = view.frontier_text("explicit_search")
    assert other.split("[Current Algorithm's Region]")[1].split("Region 2")[0].count("beam_search") >= 1


def test_v913_frozen_rules_match_the_analysis_definitions() -> None:
    from llm4ad.method.traceaad_v9_13.regions import macro_family, mechanism_tags

    corpus = {
        "tsp_construct": [
            "def f():\n    return apply_2_opt(d)",
            "def f():\n    return beam_width(x)",
            "def f():\n    return nearest(x)",
            "def f():\n    return lru_cache(f)(x)",
        ],
        "cvrp_aco": [
            "def f():\n    return sorted_customer_indices(d)",
            "def f():\n    return arctan2(a, b) + cluster_bonus(x)",
            "def f():\n    return demands[capacity]",
        ],
        "op_aco": [
            "def f(prize, distance):\n    return prize + distance",
            "def f():\n    return top_k_indices(target_scores)",
            "def f():\n    return remaining_budget - round_trip(x)",
        ],
        "online_bin_packing": [
            "def f():\n    return threshold(x)",
            "def f():\n    return np.mean(items) + percentile(x)",
            "def f():\n    return global _ + item_history(x)",
        ],
    }
    for task, codes in corpus.items():
        for code in codes:
            tags = mechanism_tags(task, code)
            assert tags == analysis_mechanism_tags(task, code)
            assert macro_family(task, tags) == analysis_macro_family(task, tags)


def test_v913_explore_before_activation_uses_plain_v97_prompt() -> None:
    method = _seeded_forest_method(treatment="fp")
    method._n_eval = FRONTIER_ACTIVATION_EVALS - 1
    request = method._prompt(method._forest.root_ids[0], Intent.EXPLORE)
    assert request.treatment == Treatment.PP.value
    assert "Searched Proxy Regions" not in request.prompt


def test_v913_fp_explore_after_activation_appends_frontier() -> None:
    method = _seeded_forest_method(treatment="fp")
    # the anchor program itself is always in the region view
    anchor = method._forest.root_ids[0]
    method._regions.record(
        _program(1, method._forest.get_program(
            method._forest.get_anchor(anchor).program_id
        ).code, -7.0, 1)
    )
    method._regions.record(_program(50, "def f():\n    return apply_2_opt(d)", -6.0, 2))
    method._regions.record(_program(51, "def b():\n    return beam_width(x)", -5.5, 3))
    method._n_eval = FRONTIER_ACTIVATION_EVALS

    request = method._prompt(method._forest.root_ids[0], Intent.EXPLORE)
    assert request.treatment == Treatment.FP.value
    assert "[Searched Proxy Regions]" in request.prompt
    assert "[Current Algorithm's Region]" in request.prompt
    assert "beam_search" in request.prompt  # other region: tags shown
    table = request.prompt.split("[Searched Proxy Regions]", 1)[1]
    assert "beam_width" not in table  # tags but no code in the table
    assert len(request.frontier_rows) == 3  # own region + two others

    refine = method._prompt(method._forest.root_ids[0], Intent.REFINE)
    assert refine.treatment == Treatment.PP.value
    assert "Searched Proxy Regions" not in refine.prompt


def test_v913_smoke_run_records_treatment_columns(tmp_path: Path) -> None:
    from llm4ad.method.traceaad_v9_13 import RunArtifacts

    method = TraceAADV913(
        llm=ScriptedLLM(),
        evaluation=ConstantEvaluation(),
        artifacts=RunArtifacts(tmp_path, console_output=False),
        budget=30,
        task_key="tsp_construct",
        checkpoint_dir=tmp_path / "checkpoints",
        seed=3,
    )
    method.run()
    summary = json.loads((tmp_path / "logs" / "summary.json").read_text())
    assert summary["status"] == "finished"
    assert summary["evaluator_call_count"] == 30
    assert summary["treatment"] == "fp"
    assert summary["treatment_counters"]["explore_pp"] + summary[
        "treatment_counters"
    ]["explore_fp"] > 0
    assert summary["frontier_activation_evals"] == FRONTIER_ACTIVATION_EVALS
    header = (tmp_path / "evaluations.csv").read_text().splitlines()[0]
    assert "treatment" in header and "reference_program_id" not in header


def test_v913_checkpoint_roundtrip_rebuilds_the_region_view(tmp_path: Path) -> None:
    method = _seeded_forest_method(treatment="fp")
    method._forest.add_program(
        code="def f():\n    return apply_2_opt(d)", fitness=-6.0, order=2
    )
    method._forest.add_program(
        code="def b():\n    return beam_width(x)", fitness=-5.5, order=3
    )
    expected = method._build_region_view()
    save_checkpoint(method, tmp_path)

    restored = _method(treatment="fp", budget=1000)
    load_checkpoint(restored, tmp_path / "latest.json")
    assert [row.program_id for row in restored._regions.frontier_rows()] == [
        row.program_id for row in expected.frontier_rows()
    ]
    assert restored._regions.frontier_text("completion_rollout") == (
        expected.frontier_text("completion_rollout")
    )
    assert restored._treatment_counters == method._treatment_counters


def test_v913_rejects_unknown_task_and_treatment() -> None:
    with pytest.raises(ValueError, match="frozen proxy rules"):
        _method(task_key="unknown_task")
    with pytest.raises(ValueError, match="unknown V9.13 treatment"):
        _method(treatment="xx")


def test_v913_runner_builds_frozen_method(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_13",
        budget=1000,
        treatment="fp",
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")
    assert isinstance(method, TraceAADV913)
    assert spec.method_name == "traceaad_v9_13"
    assert spec.n_init == 8
    assert spec.context_token_limit == 32768
    assert method._treatment == "fp"
    assert method._task_key == "tsp_construct"
    assert method._budget == 1000
    method._llm.close()

    with pytest.raises(ValueError, match="exactly eight"):
        run.make_run_spec(
            task="tsp_construct", version="v9_13", n_init=10, experiments_root=tmp_path
        )
    with pytest.raises(ValueError):
        run.make_run_spec(
            task="tsp_construct",
            version="v9_13",
            treatment="fc",
            experiments_root=tmp_path,
        )


# ---------------------------------------------------------------------------
# Stage P snapshot machinery


def _stage_p_forest_payload() -> dict:
    forest = Forest(maximize=True)
    root_program = forest.add_program(code="def root():\n    return apply_2_opt(d)", fitness=1.0, order=1)
    root = forest.add_root(program_id=root_program.id, order=1)
    child_program = forest.add_program(
        code="def child():\n    return beam_width(x)", fitness=2.0, order=2
    )
    first_id = forest.next_attempt_id()
    child = forest.add_child(
        parent_id=root.id, program_id=child_program.id, attempt_id=first_id, order=2
    )
    forest.add_attempt(
        Attempt(
            id=first_id, anchor_id=root.id, child_id=child.id,
            program_id=child_program.id,
            intent="refine", idea="i", diff=None, added=1, removed=1,
            parent_fitness=1.0, child_fitness=2.0, dq=1.0,
            outcome=Outcome.IMPROVE, kind="new", order=2, stage="search", iteration=0,
        )
    )
    second_id = forest.next_attempt_id()
    forest.add_attempt(
        Attempt(
            id=second_id, anchor_id=child.id, child_id=None, program_id=None,
            intent="explore", idea=None, diff=None, added=0, removed=0,
            parent_fitness=2.0, child_fitness=None, dq=None,
            outcome=Outcome.INVALID, kind="invalid", order=3, stage="search",
            iteration=1,
        )
    )
    return forest.to_dict()


def test_stage_p_trim_forest_counts_visits_at_the_cut() -> None:
    payload = _stage_p_forest_payload()
    trimmed = trim_forest(payload, 3)
    root, child = sorted(trimmed.anchors(), key=lambda a: a.id)
    # the order-2 response from the root is complete; the order-3 response
    # from the child has not happened yet at the cut
    assert root.n == 1 and child.n == 0
    assert [p.id for p in trimmed.programs()] == [0, 1]
    later = trim_forest(payload, 4)
    assert later.get_anchor(child.id).n == 1


def test_stage_p_interval_boundaries() -> None:
    assert EVAL_INTERVALS == ((200, 466), (467, 733), (734, 999))
    assert _interval_of(200) == 0
    assert _interval_of(466) == 0
    assert _interval_of(467) == 1
    assert _interval_of(999) == 2
    assert _interval_of(199) is None
    assert _interval_of(1000) is None


def _fake_snapshot(index: str, task: str, run: str, interval: int, q: float):
    return {
        "snapshot_index": index,
        "snapshot_id": f"{task}:{run}:it0",
        "task": task,
        "source_run": run,
        "eval_interval": interval,
        "code": "def f():\n    return apply_2_opt(d)",
        "fitness": q,
        "q": q,
        "history_text": HISTORY_TEXT,
        "anchor_family": "completion_rollout",
        "frontier_rows": [
            {"family": "explicit_search", "tags": ["beam_search"], "q": q + 1.0, "program_id": 9},
            {"family": "completion_rollout", "tags": ["two_opt"], "q": q, "program_id": 0},
        ],
        "visited_families": {"explicit_search": q + 1.0, "completion_rollout": q},
        "global_best_q": q + 1.0,
    }


def test_stage_p_schedule_blocks_share_seed_and_balance_conditions() -> None:
    snapshots = [
        _fake_snapshot(f"s{i:02d}", "tsp_construct", "run_a", i % 3, -5.0 - i * 0.1)
        for i in range(6)
    ]
    schedule = build_schedule(snapshots, replicates=3, seed=7)
    assert len(schedule) == 6 * 3 * len(CONDITIONS)
    blocks: dict[str, list[dict]] = {}
    for trial in schedule:
        blocks.setdefault(trial["block_id"], []).append(trial)
    assert len(blocks) == 18
    for trials in blocks.values():
        seeds = {trial["sampling_seed"] for trial in trials}
        assert len(seeds) == 1  # one shared seed per block
        assert sorted(trial["condition"] for trial in trials) == sorted(CONDITIONS)
    assert len(CONDITIONS) == 4
    # groups stay contiguous so one unit is served in one shard
    orders = [trial["group_order"] for trial in schedule]
    assert orders == sorted(orders)


def test_stage_p_trial_prompt_matches_v97_builder_for_pp() -> None:
    snapshot = _fake_snapshot("s00", "tsp_construct", "run_a", 0, -5.0)
    description = "Solve TSP."
    prompt = build_trial_prompt(snapshot, description, "pp_explore")
    baseline = build_v97_prompt(
        task_description=description,
        code=snapshot["code"],
        fitness=snapshot["fitness"],
        history_text=snapshot["history_text"],
        intent=Intent.EXPLORE,
        maximize=True,
    )
    assert prompt == baseline
    fp = build_trial_prompt(snapshot, description, "fp_explore")
    assert "[Searched Proxy Regions]" in fp
    assert "[Current Algorithm's Region]" in fp
    assert "two_opt" in fp and "beam_search" in fp  # own first, others with tags


def test_stage_p_next_selection_detects_winning_child() -> None:
    payload = _stage_p_forest_payload()
    snapshot = {
        "decision": {"order": 3},
        "anchor_state_id": 0,
        "code_hash": "anchorhash",
        "source_s0": 1.0,
        "archive_code_hashes": [],
    }
    from llm4ad.method.traceaad_v9_13.source import code_hash as chash

    snapshot["archive_code_hashes"] = [
        chash(program["code"]) for program in payload["programs"]
    ]
    winning = "def winner():\n    return 99"
    assert _next_selection(payload, snapshot, winning, chash(winning), 99.0) is True
    loser = "def loser():\n    return -99"
    assert _next_selection(payload, snapshot, loser, chash(loser), -99.0) is False


# ---------------------------------------------------------------------------
# Stage A fork


def _write_fake_prefix(root: Path, batch: str = "testbatch") -> Path:
    task = "tsp_construct"
    method = _method(treatment="pp", budget=PREFIX_BUDGET)
    program = method._forest.add_program(code=TEMPLATE, fitness=1.0, order=1)
    method._forest.add_root(program_id=program.id, order=1)
    method._s = 1.0
    method._initialization_complete = True
    prefix_dir = root / "experiments" / task / "traceaad_v9_13" / f"v9_13p_{batch}_{task}_rep1"
    prefix_dir.mkdir(parents=True)
    save_checkpoint(method, prefix_dir / "checkpoints")
    (prefix_dir / "logs").mkdir()
    (prefix_dir / "logs" / "summary.json").write_text(
        json.dumps({"status": "finished", "evaluator_call_count": PREFIX_BUDGET})
    )
    (prefix_dir / "run_config.json").write_text(
        json.dumps({"task": task, "method": "traceaad_v9_13", "repeat": 1})
    )
    return prefix_dir


def test_stage_a_fork_rewrites_config_and_keeps_state(tmp_path: Path) -> None:
    prefix_dir = _write_fake_prefix(tmp_path)
    unit = _unit_for_prefix(prefix_dir, treatment="fp")
    fork_prefix(unit, treatment="fp")

    prefix_state = json.loads((prefix_dir / "checkpoints" / "latest.json").read_text())
    for role_dir, treatment in ((unit.ctl_dir, "pp"), (unit.trt_dir, "fp")):
        payload = json.loads((role_dir / "checkpoints" / "latest.json").read_text())
        assert payload == prefix_state  # checkpoint copied byte-for-byte
        run_config = json.loads((role_dir / "run_config.json").read_text())
        assert run_config["method_params"]["budget"] == 1000
        assert run_config["method_params"]["treatment"] == treatment
        assert run_config["branch_treatment"] == treatment
        assert run_config["forked_from"] == prefix_dir.name

    report = audit_fork(unit)
    assert report["ok"] is True
    assert report["branches"]["control"]["state_matches_prefix"] is True
    assert report["branches"]["treatment"]["state_matches_prefix"] is True
    assert report["branches"]["treatment"]["treatment"] == "fp"
    assert report["branches"]["control"]["budget"] == 1000

    with pytest.raises(FileExistsError):
        fork_prefix(unit, treatment="fp")

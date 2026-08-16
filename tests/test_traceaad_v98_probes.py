from __future__ import annotations

import json
from pathlib import Path

from experiments.analysis.analyze_v98_mechanism_probe import _anchor_means, analyze_p3
from experiments.runners.traceaad.v98_continuation_probe import (
    DESIGN_SEED,
    Node,
    _append_formation,
    _restore_state,
    _sampling_seed_base,
    _select_p3_rows,
    _select_node,
)
from experiments.runners.traceaad.v98_mechanism_probe import CONDITIONS, build_schedule


def test_p12_schedule_pairs_all_four_conditions_with_one_seed() -> None:
    anchors = [
        {
            "anchor_id": "a0",
            "task": "tsp_construct",
            "stratum": "middle",
        }
    ]
    schedule = build_schedule(anchors, replicates=3, seed=98)
    assert len(schedule) == 12
    for replicate in range(1, 4):
        block = [row for row in schedule if row["replicate"] == replicate]
        assert {row["condition"] for row in block} == set(CONDITIONS)
        assert len({row["sampling_seed"] for row in block}) == 1


def test_p3_history_caps_at_eight_and_renumbers() -> None:
    history = "[Recent Algorithm Formation Path]"
    for index in range(10):
        history = _append_formation(
            history,
            "\n".join(
                [
                    "[History {index}] Formation step",
                    "Operator: Refine",
                    "Hypothesis: inherit H1",
                    f"Idea: step {index}",
                ]
            ),
        )
    assert history.count("Formation step") == 8
    assert "[History 1]" in history
    assert "[History 8]" in history
    assert "Idea: step 0" not in history
    assert "Idea: step 9" in history


def test_p3_protocols_use_chain_tip_or_local_anchor_score() -> None:
    nodes = {
        0: Node(0, "a", "a", 10.0, 10.0, "h", None, 0, n_refine=20),
        1: Node(1, "b", "b", 9.5, 9.5, "h", 0, 1, n_refine=0),
    }
    assert _select_node("child_chain", nodes, tip_id=1, s0=1.0).id == 1
    assert _select_node("hypothesis_level", nodes, tip_id=1, s0=1.0).id == 1
    nodes[1].n_refine = 20
    assert _select_node("hypothesis_level", nodes, tip_id=1, s0=1.0).id == 0


def test_p3_resume_reconstructs_counts_nodes_and_tip() -> None:
    unit = {
        "entry_code": "entry",
        "entry_code_hash": "entry",
        "entry_fitness": 1.0,
        "entry_q": 1.0,
        "entry_history_text": "history",
    }
    rows = [
        {
            "step": 1,
            "selected_node_id": 0,
            "child_node_id": 1,
            "candidate_code": "child",
            "candidate_code_hash": "child",
            "child_fitness": 2.0,
            "child_q": 2.0,
            "child_history_text": "child history",
            "tip_after": 1,
        },
        {
            "step": 2,
            "selected_node_id": 1,
            "child_node_id": None,
            "tip_after": 1,
        },
    ]
    nodes, tip_id, relations = _restore_state(unit, rows)
    assert tip_id == 1
    assert nodes[0].n_refine == 1
    assert nodes[1].n_refine == 1
    assert relations == {(0, "child")}


def test_p3_selects_at_most_one_valid_child_per_source_anchor() -> None:
    rows = [
        {
            "trial_id": "a0_rep2_parent_path_explore",
            "anchor_id": "a0",
            "replicate": 2,
            "condition": "parent_path_explore",
            "valid": True,
            "no_op": False,
        },
        {
            "trial_id": "a0_rep1_parent_path_explore",
            "anchor_id": "a0",
            "replicate": 1,
            "condition": "parent_path_explore",
            "valid": True,
            "no_op": False,
        },
        {
            "trial_id": "a1_rep1_parent_path_explore",
            "anchor_id": "a1",
            "replicate": 1,
            "condition": "parent_path_explore",
            "valid": True,
            "no_op": True,
        },
        {
            "trial_id": "a2_rep1_code_only_explore",
            "anchor_id": "a2",
            "replicate": 1,
            "condition": "code_only_explore",
            "valid": True,
            "no_op": False,
        },
    ]
    selected, eligible_count = _select_p3_rows(rows)
    assert eligible_count == 2
    assert [row["trial_id"] for row in selected] == [
        "a0_rep1_parent_path_explore"
    ]
    assert _sampling_seed_base(DESIGN_SEED, 71) + 5 < 2**31


def test_probe_analysis_averages_replicates_within_source_anchor() -> None:
    rows = [
        {"anchor_id": "a0", "value": 1.0},
        {"anchor_id": "a0", "value": 3.0},
        {"anchor_id": "a1", "value": 10.0},
    ]
    assert sorted(_anchor_means(rows, "value")) == [2.0, 10.0]


def test_p3_analysis_reports_paired_protocol_difference(tmp_path: Path) -> None:
    (tmp_path / "results").mkdir()
    (tmp_path / "probe_config.json").write_text(
        json.dumps(
            {
                "protocol_id": "test-p3",
                "response_count": 10,
            }
        )
    )
    (tmp_path / "units.jsonl").write_text(
        json.dumps(
            {
                "unit_id": "u0",
                "task": "tsp_construct",
                "entry_q": 0.0,
                "parent_q": 2.0,
            }
        )
        + "\n"
    )
    rows = []
    for protocol, final_q in (("child_chain", 1.0), ("hypothesis_level", 3.0)):
        for step in range(1, 6):
            rows.append(
                {
                    "continuation_id": f"u0_{protocol}",
                    "unit_id": "u0",
                    "task": "tsp_construct",
                    "protocol": protocol,
                    "step": step,
                    "valid": True,
                    "child_q": final_q if step == 1 else final_q,
                }
            )
    (tmp_path / "results" / "shard_00.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    summary = analyze_p3(tmp_path)

    paired = summary["paired_protocol_contrasts"]["tsp_construct:H1"]
    assert paired["n"] == 1
    assert paired["hypothesis_minus_chain_internal_gain"]["mean"] == 2.0

import json

from experiments.traceaad_e2_b.e2b import (
    ARMS,
    paired,
    trajectory_metrics,
)


def test_q2_keeps_invalid_descendants_in_unconditional_estimand():
    row = {
        "anchor_fitness": -10.0,
        "step1": {"evaluation": {"fitness": None}},
        "step2": None,
    }
    result = trajectory_metrics(row)
    assert result["q1"] == 0
    assert result["q2"] == 0
    assert result["relative_q2"] == 0
    assert result["continuation_gain"] is None


def test_q2_uses_best_point_but_continuation_keeps_signed_change():
    row = {
        "anchor_fitness": -10.0,
        "step1": {"evaluation": {"fitness": -12.0}},
        "step2": {"evaluation": {"fitness": -9.0}},
    }
    result = trajectory_metrics(row)
    assert result["q1"] == 0
    assert result["q2"] == 1
    assert result["continuation_gain"] == 3


def test_paired_contrast_is_pivot_refine_minus_refine_refine():
    rows = [
        {"anchor_id": "a", "arm": "RR", "metric": 1.0},
        {"anchor_id": "a", "arm": "PR", "metric": 4.0},
        {"anchor_id": "b", "arm": "RR", "metric": 3.0},
        {"anchor_id": "b", "arm": "PR", "metric": 2.0},
    ]
    result = paired(rows, "metric")
    assert set(ARMS) == {"RR", "PR"}
    assert result["paired_anchors"] == 2
    assert result["PR_minus_RR_mean"] == 1.0
    assert result["PR_better"] == 1
    assert result["RR_better"] == 1


def test_frozen_schedule_is_complete_and_state_definition_is_enforced():
    root = "experiments/_logs/traceaad_e2_b_20260907"
    anchors = [json.loads(line) for line in open(f"{root}/anchors.jsonl")]
    schedule = [json.loads(line) for line in open(f"{root}/schedule.jsonl")]
    assert len(anchors) == 18
    assert len(schedule) == 36
    for anchor in anchors:
        if anchor["state"] == "experienced":
            assert anchor["development_attempts"] >= 3
            assert anchor["recent_parent_or_frontier_gains"] == 0
            assert anchor["valid_behavior_children"] >= 2
            assert anchor["median_child_movement"] <= anchor["run_move_median"]
            assert anchor["child_revisit_rate"] >= 0.5
        else:
            assert anchor["development_attempts"] == 0
    by_anchor = {}
    for trial in schedule:
        by_anchor.setdefault(trial["anchor_id"], []).append(trial)
    assert all({trial["arm"] for trial in trials} == {"RR", "PR"} for trials in by_anchor.values())
    assert all(len({trial["step1_seed"] for trial in trials}) == 1 for trials in by_anchor.values())
    assert all(len({trial["step2_seed"] for trial in trials}) == 1 for trials in by_anchor.values())

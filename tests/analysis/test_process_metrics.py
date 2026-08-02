"""Regression tests for the shared process-analysis metric definitions."""

from __future__ import annotations

import pandas as pd

from experiments.analysis.analyze_traceaad_process import (
    best_update_event_stats,
    sample_level_best_curve,
    sample_level_best_stats,
    task_zscore,
)


def test_sample_breakthroughs_exclude_equal_score_ties_and_use_actual_windows():
    samples = [
        {"sample_order": 1, "score": 1.0},
        {"sample_order": 2, "score": 2.0},
        {"sample_order": 3, "score": 2.0},  # equal fitness, not a breakthrough
        {"sample_order": 11, "score": 1.5},
        {"sample_order": 12, "score": 3.0},
    ]

    stats = sample_level_best_stats(samples)

    assert stats["breakthrough_orders"] == [2, 12]
    assert stats["breakthrough_windows"] == {1, 2}
    assert stats["n_windows"] == 2
    assert stats["breakthrough_rate_w10"] == 1.0
    assert sample_level_best_curve(samples) == [1.0, *([2.0] * 10), 3.0]


def test_best_update_tie_shorter_is_a_separate_event_audit_metric():
    events = [
        {
            "event": "best_updated",
            "previous_best_node_id": None,
            "update_reason": "strict_fitness",
        },
        {
            "event": "best_updated",
            "previous_best_node_id": 1,
            "update_reason": "strict_fitness",
        },
        {
            "event": "best_updated",
            "previous_best_node_id": 2,
            "update_reason": "tie_shorter",
        },
        {"event": "best_updated", "previous_best_node_id": 3},
    ]

    assert best_update_event_stats(events) == {
        "best_update_event_count": 3,
        "strict_fitness_event_count": 1,
        "tie_shorter_count": 1,
        "unclassified_best_update_count": 1,
    }


def test_task_zscore_isolated_by_task_and_constant_groups_are_zero():
    frame = pd.DataFrame(
        {
            "task": ["a", "a", "b", "b"],
            "score": [1.0, 3.0, 5.0, 5.0],
        }
    )

    z = task_zscore(frame, "score")

    assert z.iloc[0] == -z.iloc[1]
    assert z.iloc[2] == 0.0
    assert z.iloc[3] == 0.0

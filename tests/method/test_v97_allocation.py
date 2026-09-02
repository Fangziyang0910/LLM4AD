"""Tests for the V9.7 same-snapshot allocation diagnostic."""

from __future__ import annotations

import pytest

from experiments.traceaad_v9_7.analyze import (
    aggregate_runs,
    analyze_anchor_event,
    analyze_route_event,
    phase_for,
    quantile,
    summarize_interventions,
)


def test_route_intervention_uses_bonus_difference_not_absolute_bonus() -> None:
    event = {
        "event": "route_selected",
        "iteration": 7,
        "selected_root_state_id": 1,
        "routes": [
            {
                "root_state_id": 0,
                "best_q": 100.0,
                "n": 400,
                "optimism": 0.5,
                "score": 100.5,
            },
            {
                "root_state_id": 1,
                "best_q": 98.0,
                "n": 10,
                "optimism": 3.0,
                "score": 101.0,
            },
        ],
    }

    result = analyze_route_event(event)

    assert result["intervened"]
    assert result["quality_winner_id"] == 0
    assert result["selected_id"] == 1
    assert result["quality_gap"] == 2.0
    assert result["optimism_advantage"] == 2.5
    assert result["margin"] == 0.5
    assert result["critical_multiplier"] == pytest.approx(0.8)


def test_route_replay_preserves_less_consumed_quality_tie_break() -> None:
    event = {
        "event": "route_selected",
        "iteration": 0,
        "selected_root_state_id": 1,
        "routes": [
            {
                "root_state_id": 0,
                "best_q": 5.0,
                "n": 4,
                "optimism": 0.0,
                "score": 5.0,
            },
            {
                "root_state_id": 1,
                "best_q": 5.0,
                "n": 1,
                "optimism": 0.0,
                "score": 5.0,
            },
        ],
    }

    result = analyze_route_event(event)

    assert result["quality_winner_id"] == 1
    assert not result["intervened"]
    assert result["critical_multiplier"] is None


def test_anchor_intervention_replays_quality_only_inside_selected_route() -> None:
    event = {
        "event": "anchor_selected",
        "iteration": 3,
        "selected_state_id": 11,
        "states": [
            {
                "state_id": 10,
                "q": 100.0,
                "n": 10,
                "optimism": 0.2,
                "score": 100.2,
                "creation_order": 1,
            },
            {
                "state_id": 11,
                "q": 99.8,
                "n": 0,
                "optimism": 1.0,
                "score": 100.8,
                "creation_order": 2,
            },
        ],
    }

    result = analyze_anchor_event(event)

    assert result["intervened"]
    assert result["quality_winner_id"] == 10
    assert result["selected_id"] == 11


def test_phase_partition_and_intervention_summary_are_deterministic() -> None:
    rows = []
    for index in range(8):
        rows.append(
            {
                "intervened": index in {0, 3, 7},
                "phase": phase_for(index, 8),
            }
        )

    summary = summarize_interventions(rows)

    assert [phase_for(index, 8) for index in range(8)] == [
        "early",
        "early",
        "early",
        "middle",
        "middle",
        "middle",
        "late",
        "late",
    ]
    assert summary["decisions"] == 8
    assert summary["interventions"] == 3
    assert summary["by_phase"]["early"]["interventions"] == 1
    assert summary["by_phase"]["middle"]["interventions"] == 1
    assert summary["by_phase"]["late"]["interventions"] == 1


def test_quantile_and_task_aggregation_use_decision_weighting() -> None:
    assert quantile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    runs = [
        {
            "task_label": "TSP",
            "bootstrap_zero_delta_count": 1,
            "route": {
                "decisions": 2,
                "interventions": 1,
                "intervention_rate": 0.5,
                "by_phase": {
                    "early": {"decisions": 1, "interventions": 1},
                    "middle": {"decisions": 1, "interventions": 0},
                    "late": {"decisions": 0, "interventions": 0},
                },
                "max_route_share": 0.75,
                "hhi": 0.625,
            },
            "anchor": {
                "decisions": 2,
                "interventions": 2,
                "intervention_rate": 1.0,
                "by_phase": {
                    "early": {"decisions": 1, "interventions": 1},
                    "middle": {"decisions": 1, "interventions": 1},
                    "late": {"decisions": 0, "interventions": 0},
                },
            },
        },
        {
            "task_label": "TSP",
            "bootstrap_zero_delta_count": 0,
            "route": {
                "decisions": 8,
                "interventions": 0,
                "intervention_rate": 0.0,
                "by_phase": {
                    "early": {"decisions": 3, "interventions": 0},
                    "middle": {"decisions": 3, "interventions": 0},
                    "late": {"decisions": 2, "interventions": 0},
                },
                "max_route_share": 1.0,
                "hhi": 1.0,
            },
            "anchor": {
                "decisions": 8,
                "interventions": 0,
                "intervention_rate": 0.0,
                "by_phase": {
                    "early": {"decisions": 3, "interventions": 0},
                    "middle": {"decisions": 3, "interventions": 0},
                    "late": {"decisions": 2, "interventions": 0},
                },
            },
        },
    ]

    aggregate = aggregate_runs(runs)

    assert aggregate["overall"]["route"]["intervention_rate"] == 0.1
    assert aggregate["overall"]["anchor"]["intervention_rate"] == 0.2
    assert aggregate["overall"]["runs_with_zero_bootstrap_delta"] == 1
    assert aggregate["tasks"]["TSP"]["max_route_share_min"] == 0.75
    assert aggregate["tasks"]["TSP"]["max_route_share_max"] == 1.0

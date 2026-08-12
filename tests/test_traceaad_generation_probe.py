from __future__ import annotations

from experiments.runners.traceaad.generation_probe import (
    CONDITIONS,
    STRATA,
    build_prompt_pair,
    build_schedule,
    render_concise_history,
)
from llm4ad.method.traceaad_v9_5.source import text_hash


def _anchor(index: int = 0) -> dict:
    return {
        "anchor_id": f"anchor_{index:03d}",
        "snapshot_id": f"snapshot_{index}",
        "task": "op_aco",
        "stratum": STRATA[index % 3],
        "fitness": 12.5,
        "q": 12.5,
        "code": "def heuristics(x):\n    return x\n",
        "history": [
            {
                "source": "Formation",
                "idea": "add distance normalization",
                "result": "improve",
                "parent_fitness": 11.0,
                "child_fitness": 12.0,
            },
            {
                "source": "Direct",
                "idea": "increase the exponent",
                "result": "regress",
                "parent_fitness": 12.5,
                "child_fitness": 12.1,
            },
        ],
    }


def test_prompt_pair_differs_only_by_history_block() -> None:
    anchor = _anchor()
    pair = build_prompt_pair(anchor, "Improve a heuristic.")

    assert "[Current Fitness]\n12.5" in pair.no_history
    assert "[Concise Search History]" not in pair.no_history
    assert "[Concise Search History]" in pair.concise_history
    assert pair.concise_history.replace(pair.history_block + "\n\n", "", 1) == pair.no_history
    assert text_hash(pair.canonical_without_history) == text_hash(pair.no_history)


def test_concise_history_contains_only_frozen_fields() -> None:
    rendered = render_concise_history(_anchor())

    assert "Source: Formation" in rendered
    assert "Idea: add distance normalization" in rendered
    assert "Result: regress" in rendered
    assert "Fitness: 12.5 -> 12.1" in rendered
    assert "diff" not in rendered.lower()
    assert "loc" not in rendered.lower()


def test_schedule_is_paired_seeded_and_ab_ba_balanced() -> None:
    anchors = []
    for index in range(12):
        anchor = _anchor(index)
        anchor["task"] = "op_aco" if index < 6 else "tsp_construct"
        anchors.append(anchor)
    schedule = build_schedule(anchors, replicates=2, seed=17)

    assert len(schedule) == len(anchors) * 2 * len(CONDITIONS)
    pairs = {}
    for row in schedule:
        pairs.setdefault(row["pair_id"], []).append(row)
    assert all(len(rows) == 2 for rows in pairs.values())
    assert all({row["condition"] for row in rows} == set(CONDITIONS) for rows in pairs.values())
    assert all(len({row["sampling_seed"] for row in rows}) == 1 for rows in pairs.values())
    for task in {row["task"] for row in schedule}:
        for stratum in STRATA:
            blocks = [
                rows
                for rows in pairs.values()
                if rows[0]["task"] == task and rows[0]["stratum"] == stratum
            ]
            first = [min(rows, key=lambda row: row["within_pair_order"])["condition"] for rows in blocks]
            assert first.count("no_history") == first.count("concise_history")

from __future__ import annotations

from experiments.runners.traceaad.compact_change_probe import (
    CONDITIONS,
    build_compact_prompt_pair,
    build_schedule,
    compact_actual_change,
)
from experiments.runners.traceaad.generation_probe import STRATA


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
                "idea": "normalize the input",
                "result": "improve",
                "parent_fitness": 11.0,
                "child_fitness": 12.0,
                "compact_actual_change": (
                    "Diff size: +1/-1 lines. Removed examples: `return x`. "
                    "Added examples: `return x / scale`."
                ),
                "actual_diff_available": True,
            },
            {
                "source": "Direct",
                "idea": "increase the exponent",
                "result": "regress",
                "parent_fitness": 12.5,
                "child_fitness": 12.1,
                "compact_actual_change": "No executable code change was recorded.",
                "actual_diff_available": False,
            },
        ],
    }


def test_compact_change_is_deterministic_and_not_a_raw_diff() -> None:
    candidate = {
        "actual_diff": """--- parent.py
+++ candidate.py
@@ -1,4 +1,5 @@
-    # old explanation
-    score = distance
-    return score
+    # new explanation
+    score = distance ** 2
+    penalty = demand / capacity
+    return score * penalty
""",
        "diff_statistics": {"added_lines": 4, "removed_lines": 3},
    }

    rendered = compact_actual_change(candidate)

    assert rendered == compact_actual_change(candidate)
    assert "Diff size: +4/-3 lines." in rendered
    assert "score = distance" in rendered
    assert "return score * penalty" in rendered
    assert "old explanation" not in rendered
    assert "@@" not in rendered
    assert len(rendered) <= 520


def test_prompt_pair_differs_only_by_compact_change_block() -> None:
    pair = build_compact_prompt_pair(_anchor(), "Improve a heuristic.")

    assert "[Concise Search History]" in pair.concise_history
    assert "[Compact Actual Changes]" not in pair.concise_history
    assert "[Compact Actual Changes]" in pair.compact_actual_change
    assert (
        pair.compact_actual_change.replace(
            "\n\n" + pair.compact_change_block, "", 1
        )
        == pair.concise_history
    )


def test_schedule_is_paired_and_balanced() -> None:
    anchors = []
    for index in range(12):
        anchor = _anchor(index)
        anchor["task"] = "op_aco" if index < 6 else "tsp_construct"
        anchors.append(anchor)
    schedule = build_schedule(anchors, replicates=2, seed=23)

    pairs: dict[str, list[dict]] = {}
    for row in schedule:
        pairs.setdefault(row["pair_id"], []).append(row)
    assert len(schedule) == len(anchors) * 2 * len(CONDITIONS)
    assert all({row["condition"] for row in rows} == set(CONDITIONS) for rows in pairs.values())
    assert all(len({row["sampling_seed"] for row in rows}) == 1 for rows in pairs.values())
    for task in {row["task"] for row in schedule}:
        for stratum in STRATA:
            blocks = [
                rows
                for rows in pairs.values()
                if rows[0]["task"] == task and rows[0]["stratum"] == stratum
            ]
            first = [
                min(rows, key=lambda row: row["within_pair_order"])["condition"]
                for rows in blocks
            ]
            assert first.count(CONDITIONS[0]) == first.count(CONDITIONS[1])

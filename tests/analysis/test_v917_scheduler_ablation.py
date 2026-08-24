from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.analysis.analyze_v917_scheduler_ablation import analyze_run


def test_v917_scheduler_analysis_uses_primary_slots_and_block_frontiers(
    tmp_path: Path,
) -> None:
    fieldnames = [
        "eval_count",
        "intent",
        "mode",
        "hypothesis_id",
        "block_id",
        "status",
        "fitness",
        "attempt_kind",
    ]
    rows = [
        ["1", "", "root", "1", "", "ok", "1", "initial"],
        ["2", "refine", "development", "1", "1", "runtime_error", "", "initial"],
        ["2", "refine", "development", "1", "1", "ok", "2", "repair"],
        ["3", "refine", "development", "1", "1", "ok", "3", "initial"],
        ["4", "explore", "discovery", "2", "", "ok", "4", "initial"],
    ]
    with (tmp_path / "evaluations.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fieldnames)
        writer.writerows(rows)
    events = [
        {
            "event": "block_finish",
            "block_id": 1,
            "block_kind": "development",
            "hypothesis_id": 1,
            "cycle": 1,
            "sweep": 2,
            "gain": 2,
        }
    ]
    (tmp_path / "mechanism_events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    metrics = analyze_run(tmp_path)
    assert metrics["current_primary_slots"] == 4
    assert metrics["intent_slots"] == {"root": 1, "refine": 2, "explore": 1}
    assert metrics["search_best"] == 4.0
    assert metrics["adaptive_extra_blocks"] == 1
    assert metrics["extra_blocks_advancing_hypothesis_frontier"] == 1
    assert metrics["extra_blocks_advancing_global_frontier"] == 1

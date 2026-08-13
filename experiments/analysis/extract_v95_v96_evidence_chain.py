"""Extract per-generation evidence-chain records for the V9.5 vs V9.6 comparison.

Reads candidates/decisions/llm_calls JSONL from the formal V9.5 (20260811_171029)
and V9.6 (20260812_191011) runs, joins each anchor-step generation with the
history-construction audit event that produced its prompt, and writes two CSVs:

- candidates.csv: one row per candidate (generation attempt), with outcome,
  delta fitness vs anchor, edit size, cumulative real evaluator calls,
  best-so-far, and the joined history composition at generation time.
- history_events.csv: one row per evidence_built / history_built event, with
  pool and shown composition (formation/direct counts, outcome shares, tokens).

Outputs go to experiments/analysis/logs_v95_v96_evidence_chain/ (local only).

Usage: uv run python -m experiments.analysis.extract_v95_v96_evidence_chain
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "experiments" / "analysis" / "logs_v95_v96_evidence_chain"

TASKS = {
    "op_aco": "op",
    "online_bin_packing": "obp",
    "tsp_construct": "tsp",
    "cvrp_aco": "cvrp",
}

V95_BATCH = "v9_5_20260811_171029"
V96_BATCH = "v9_6_20260812_191011"


def discover_runs() -> list[dict]:
    runs = []
    for task, short in TASKS.items():
        for version, batch, method_dir in (
            ("v9.5", V95_BATCH, "traceaad_v9_5"),
            ("v9.6", V96_BATCH, "traceaad_v9_6"),
        ):
            for rep in (1, 2, 3):
                run_dir = (
                    REPO / "experiments" / task / method_dir / f"{batch}_{short}_rep{rep}"
                )
                if not run_dir.exists():
                    continue
                summary_path = run_dir / "logs" / "summary.json"
                status = None
                if summary_path.exists():
                    status = json.loads(summary_path.read_text()).get("status")
                runs.append(
                    {
                        "task": task,
                        "version": version,
                        "rep": rep,
                        "run_dir": run_dir,
                        "run_name": run_dir.name,
                        "status": status or "running",
                    }
                )
    return runs


def iter_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def outcome_category(attempt_kind: str | None, direct_outcome: str | None) -> str:
    """Collapse attempt_kind x direct_outcome into one generation-outcome label."""
    if attempt_kind == "root_new":
        return "root_new"
    if attempt_kind == "invalid":
        return "invalid"
    if attempt_kind == "no_op":
        return "no_op"
    if attempt_kind in ("repeated_duplicate", "ancestral_return"):
        return "duplicate"
    return direct_outcome or "unknown"


def extract_run(run: dict) -> tuple[list[dict], list[dict]]:
    art = run["run_dir"] / "artifacts"

    # --- llm_calls: sample_order -> token counts -------------------------------
    tokens: dict[int, tuple[int | None, int | None]] = {}
    for d in iter_jsonl(art / "llm_calls.jsonl"):
        tokens[d["sample_order"]] = (d.get("prompt_tokens"), d.get("response_tokens"))

    # --- candidates -------------------------------------------------------------
    cands: list[dict] = []
    state_q: dict[int, float] = {}
    attempt_outcome: dict[int, str] = {}
    for d in iter_jsonl(art / "candidates.jsonl"):
        ds = d.get("diff_statistics") or {}
        row = {
            "attempt_id": d.get("attempt_id"),
            "sample_order": d.get("sample_order"),
            "iteration": d.get("iteration"),
            "stage": d.get("stage"),
            "attempt_kind": d.get("attempt_kind"),
            "direct_outcome": d.get("direct_outcome"),
            "status": d.get("status"),
            "score": d.get("score"),
            "parent_node_id": d.get("parent_node_id"),
            "child_state_id": d.get("child_state_id"),
            "evaluator_called": bool(d.get("evaluator_called")),
            "added_lines": ds.get("added_lines"),
            "removed_lines": ds.get("removed_lines"),
            "changed_lines": ds.get("changed_lines"),
        }
        cands.append(row)
        if row["child_state_id"] is not None and row["score"] is not None:
            state_q[row["child_state_id"]] = row["score"]
        if row["attempt_id"] is not None:
            attempt_outcome[row["attempt_id"]] = outcome_category(
                row["attempt_kind"], row["direct_outcome"]
            )
    cands.sort(key=lambda r: r["sample_order"])

    # --- decisions: bind history events to the following attempt ----------------
    hist_key = "history_built" if run["version"] == "v9.6" else "evidence_built"
    hist_events: list[dict] = []
    attempt_to_hist: dict[int, int] = {}
    pending_hist: int | None = None
    last_anchor: dict | None = None
    for d in iter_jsonl(art / "decisions.jsonl"):
        ev = d.get("event")
        if ev == "anchor_selected":
            last_anchor = d
        elif ev == hist_key:
            form_pool = d.get("formation_pool_ids") or []
            dir_pool = d.get("direct_pool_ids") or []
            sel_form = d.get("selected_formation_ids") or []
            sel_dir = d.get("selected_direct_ids") or []

            def count(ids: list[int], label: str) -> int:
                return sum(attempt_outcome.get(i) == label for i in ids)

            if run["version"] == "v9.6":
                truncated = 1 if (d.get("dropped_for_context") or 0) > 0 else 0
                dropped = d.get("dropped_for_context") or 0
            else:
                truncated = 1 if (
                    (d.get("truncated_attempt_ids") or [])
                    or (d.get("diff_excerpt_chars") or 1200) < 1200
                ) else 0
                dropped = len(d.get("truncated_attempt_ids") or [])
            anchor_iter = last_anchor.get("iteration") if last_anchor else None
            anchor_match = bool(
                last_anchor
                and last_anchor.get("selected_state_id") == d.get("anchor_state_id")
            )
            hist_events.append(
                {
                    "hist_idx": len(hist_events),
                    "anchor_state_id": d.get("anchor_state_id"),
                    "iteration": anchor_iter if anchor_match else None,
                    "n_formation_pool": len(form_pool),
                    "n_direct_pool": len(dir_pool),
                    "n_formation_shown": len(sel_form),
                    "n_direct_shown": len(sel_dir),
                    "n_shown": len(sel_form) + len(sel_dir),
                    "n_dir_pool_improve": count(dir_pool, "improve"),
                    "n_dir_pool_regress": count(dir_pool, "regress"),
                    "n_dir_shown_improve": count(sel_dir, "improve"),
                    "n_dir_shown_regress": count(sel_dir, "regress"),
                    "n_form_pool_improve": count(form_pool, "improve"),
                    "n_form_pool_regress": count(form_pool, "regress"),
                    "n_form_shown_improve": count(sel_form, "improve"),
                    "n_form_shown_regress": count(sel_form, "regress"),
                    "truncated": truncated,
                    "n_dropped": dropped,
                    "hist_prompt_tokens": d.get("prompt_tokens"),
                }
            )
            pending_hist = hist_events[-1]["hist_idx"]
        elif ev == "attempt_finalized":
            if pending_hist is not None and d.get("attempt_id") is not None:
                attempt_to_hist[d["attempt_id"]] = pending_hist
            pending_hist = None

    # --- assemble candidate rows -------------------------------------------------
    hist_by_idx = {h["hist_idx"]: h for h in hist_events}
    cum_calls = 0
    best_so_far: float | None = None
    out_rows: list[dict] = []
    for c in cands:
        if c["evaluator_called"]:
            cum_calls += 1
        anchor_q = (
            state_q.get(c["parent_node_id"]) if c["parent_node_id"] is not None else None
        )
        delta = (
            c["score"] - anchor_q
            if (c["score"] is not None and anchor_q is not None)
            else None
        )
        is_new_best = 0
        if c["score"] is not None and c["status"] == "ok":
            if best_so_far is None or c["score"] > best_so_far:
                best_so_far = c["score"]
                is_new_best = 1
        h = hist_by_idx.get(attempt_to_hist.get(c["attempt_id"], -1))
        tok = tokens.get(c["sample_order"], (None, None))
        out_rows.append(
            {
                "task": run["task"],
                "version": run["version"],
                "rep": run["rep"],
                "run_name": run["run_name"],
                "run_status": run["status"],
                "sample_order": c["sample_order"],
                "iteration": c["iteration"],
                "stage": c["stage"],
                "outcome": outcome_category(c["attempt_kind"], c["direct_outcome"]),
                "status": c["status"],
                "score": c["score"],
                "anchor_q": anchor_q,
                "delta": delta,
                "evaluator_called": int(c["evaluator_called"]),
                "cum_eval_calls": cum_calls,
                "best_so_far": best_so_far,
                "is_new_best": is_new_best,
                "added_lines": c["added_lines"],
                "removed_lines": c["removed_lines"],
                "changed_lines": c["changed_lines"],
                "prompt_tokens": tok[0],
                "response_tokens": tok[1],
                "n_formation_pool": h["n_formation_pool"] if h else None,
                "n_direct_pool": h["n_direct_pool"] if h else None,
                "n_formation_shown": h["n_formation_shown"] if h else None,
                "n_direct_shown": h["n_direct_shown"] if h else None,
                "hist_truncated": h["truncated"] if h else None,
            }
        )

    for h in hist_events:
        h.update(
            {
                "task": run["task"],
                "version": run["version"],
                "rep": run["rep"],
                "run_name": run["run_name"],
                "run_status": run["status"],
            }
        )
    return out_rows, hist_events


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs = discover_runs()
    all_cands: list[dict] = []
    all_hists: list[dict] = []
    for run in runs:
        rows, hists = extract_run(run)
        all_cands.extend(rows)
        all_hists.extend(hists)
        n_anchor = sum(1 for r in rows if r["stage"] not in ("root_generation",))
        print(
            f"{run['run_name']:44s} status={run['status']:22s} "
            f"candidates={len(rows):5d} anchor_steps={n_anchor:5d} "
            f"hist_events={len(hists):5d}"
        )

    for name, rows in (("candidates.csv", all_cands), ("history_events.csv", all_hists)):
        path = OUT_DIR / name
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

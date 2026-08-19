"""V9.9 (batch 20260816_154200) process diagnostics over logs/events.jsonl.

Reproduces the numbers cited in docs/analysis/TraceAAD-V9.9机制诊断.md:

  A. allocation geometry by phase (entropy, top5/10/20 mu, selected rank, ...)
  B. operator share by phase + P(E|a) + C_R activation among refine selections
  C. outcome accounting by operator (improve/plateau/regress/invalid)
  D. anchor life-cycle (born-dead, explore-born starvation) and best-event timeline
  E. lineage analysis (winning path composition, lineage selection share, anchor age)

V9.7 comparison (route share, winning path depth) reads artifacts/decisions.jsonl
and artifacts/candidates.jsonl of batch 20260814_150927 via --v97.

Runs still in progress (no logs/summary.json) are handled by counting
evaluations.csv lines.
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

V99_TASKS = [
    ("tsp", "tsp_construct"),
    ("cvrp", "cvrp_aco"),
    ("op", "op_aco"),
    ("obp", "online_bin_packing"),
]


def phase(eval_count):
    if eval_count <= 333:
        return "early"
    if eval_count <= 666:
        return "mid"
    return "late"


def load_run(run_dir):
    reqs, resps, order = {}, {}, []
    with open(f"{run_dir}/logs/events.jsonl") as f:
        for line in f:
            e = json.loads(line)
            if e["event"] == "request_prepared":
                reqs[e["response_id"]] = e
            elif e["event"] == "response_finalized":
                resps[e["response_id"]] = e
                order.append(e["response_id"])
    summary_path = f"{run_dir}/logs/summary.json"
    if os.path.exists(summary_path):
        summary = json.load(open(summary_path))
    else:
        summary = {"evaluator_call_count": 0}
        csv = f"{run_dir}/evaluations.csv"
        if os.path.exists(csv):
            with open(csv) as f:
                summary["evaluator_call_count"] = max(1, sum(1 for _ in f) - 1)
    return reqs, resps, order, summary


def search_responses(resps, order):
    for rid in order:
        r = resps[rid]
        if r.get("stage") == "search":
            yield rid, r


def v99_runs():
    for task, name in V99_TASKS:
        for rd in sorted(glob.glob(f"{BASE}/{name}/traceaad_v9_9/v9_9_20260816_154200_*_rep*")):
            rep = os.path.basename(rd).split("_rep")[1]
            yield task, rep, rd


def section_allocation_geometry():
    print("=" * 100)
    print("A. ALLOCATION GEOMETRY by phase (mean over decisions in phase)")
    print(f"{'task/rep':10} {'phase':6} {'entropy':>8} {'top5mu':>8} {'top10mu':>8} "
          f"{'top10uniq':>9} {'#anchors':>9} {'selrank':>8} {'rank<=5':>8} {'multi':>6}")
    for task, rep, rd in v99_runs():
        reqs, resps, order, summary = load_run(rd)
        agg = defaultdict(lambda: defaultdict(list))
        for rid, r in search_responses(resps, order):
            q = reqs.get(rid)
            if q is None:
                continue
            d = q["selection"]["diagnostics"]
            ph = phase(r["eval_count"])
            for k in ("selection_entropy", "top5_mu", "top10_mu", "top10_unique_programs",
                      "n_anchors", "selected_rank", "selected_program_multiplicity"):
                if d.get(k) is not None:
                    agg[ph][k].append(d[k])
            if d.get("selected_rank") is not None:
                agg[ph]["rank_le5"].append(1 if d["selected_rank"] <= 5 else 0)
        for ph in ("early", "mid", "late"):
            a = agg.get(ph)
            if not a:
                continue
            m = lambda k: sum(a[k]) / len(a[k])
            print(f"{task}/{rep:<7} {ph:6} {m('selection_entropy'):8.3f} {m('top5_mu'):8.3f} "
                  f"{m('top10_mu'):8.3f} {m('top10_unique_programs'):9.2f} {m('n_anchors'):9.1f} "
                  f"{m('selected_rank'):8.2f} {m('rank_le5')*100:7.1f}% "
                  f"{m('selected_program_multiplicity'):6.3f}")


def section_operators():
    print()
    print("=" * 100)
    print("B. OPERATOR SHARE by phase, P(E|a) at selection, C_R activation among refine selections")
    print(f"{'task/rep':10} {'E_early':>8} {'E_mid':>8} {'E_late':>8} {'piE(e/m/l)':>14} "
          f"{'cR>0/Rsel(e/m/l)':>18}")
    for task, rep, rd in v99_runs():
        reqs, resps, order, _ = load_run(rd)
        ops = defaultdict(Counter)
        pi_e = defaultdict(list)
        c_r_pos, c_r_tot, c_r_val = defaultdict(int), defaultdict(int), defaultdict(list)
        for rid, r in search_responses(resps, order):
            q = reqs.get(rid)
            d = q["selection"]["diagnostics"] if q else {}
            ph = phase(r["eval_count"])
            ops[ph][r["intent"]] += 1
            if d:
                pi_e[ph].append(d.get("selected_pi_explore") or 0)
                if r["intent"] == "refine":
                    c_r_tot[ph] += 1
                    c = d.get("selected_c_refine") or 0
                    c_r_val[ph].append(c)
                    if c > 0:
                        c_r_pos[ph] += 1
        parts = [f"{task}/{rep:<9}"]
        for ph in ("early", "mid", "late"):
            c = ops[ph]
            parts.append(f"{c['explore'] / max(1, c['explore'] + c['refine']) * 100:7.1f}%")
        for ph in ("early", "mid", "late"):
            parts.append(f"{sum(pi_e[ph]) / max(1, len(pi_e[ph])):5.3f}")
        parts.append("  " + " | ".join(f"{c_r_pos[ph]}/{c_r_tot[ph]}" for ph in ("early", "mid", "late")))
        parts.append("  meanC_R>0: " + ",".join(
            f"{ph}:{sum(v for v in c_r_val[ph] if v > 0) / max(1, sum(1 for v in c_r_val[ph] if v > 0)):.3f}"
            for ph in ("early", "mid", "late")))
        print("".join(parts))


def section_outcomes():
    print()
    print("=" * 100)
    print("C. OUTCOME ACCOUNTING by operator (search stage, all phases)")
    print(f"{'task/rep':10} {'op':7} {'improve':>8} {'plateau':>8} {'regress':>8} {'invalid':>8} {'tot':>6}")
    for task, rep, rd in v99_runs():
        reqs, resps, order, _ = load_run(rd)
        for op in ("refine", "explore"):
            c = Counter()
            for _, r in search_responses(resps, order):
                if r["intent"] == op:
                    c[r["outcome"]] += 1
            tot = sum(c.values())
            if tot:
                print(f"{task}/{rep:<7} {op:7} {c['improve']:8} {c['plateau']:8} {c['regress']:8} "
                      f"{c['invalid']:8} {tot:6}")


def section_lifecycle():
    print()
    print("=" * 100)
    print("D. ANCHOR LIFE-CYCLE & IMPROVEMENT TIMELINE")
    print(f"{'task/rep':10} {'#born':>6} {'born-dead':>9} {'E-born':>7} {'E-dead':>7} "
          f"{'#best':>6} {'best ops':>16} {'lastBest':>8} {'bestAtResp':>10}")
    for task, rep, rd in v99_runs():
        reqs, resps, order, summary = load_run(rd)
        born, selected, best_events = {}, set(), []
        for rid, r in search_responses(resps, order):
            if r["kind"] in ("new", "duplicate") and r["child_id"] is not None:
                born[r["child_id"]] = r["intent"]
            if reqs.get(rid) is not None:
                selected.add(reqs[rid]["anchor_id"])
            if r.get("is_new_best"):
                best_events.append(r)
        e_born = [a for a, op in born.items() if op == "explore"]
        ops = Counter(b["intent"] for b in best_events)
        best_ops = ",".join(f"{k}:{v}" for k, v in ops.items())
        print(f"{task}/{rep:<7} {len(born):6} {sum(1 for a in born if a not in selected):9} "
              f"{len(e_born):7} {sum(1 for a in e_born if a not in selected):7} "
              f"{len(best_events):6} {best_ops:>16} "
              f"{max((b['eval_count'] for b in best_events), default=0):8} "
              f"{str(summary.get('best_response_order')):>10}")


def section_lineage():
    print()
    print("=" * 100)
    print("E. LINEAGE ANALYSIS (winning path composition, budget share, anchor age)")
    print(f"{'task/rep':10} {'pathLen':>7} {'E-edges':>8} {'R-edges':>8} {'lineageSel%':>11} "
          f"{'age<10':>7} {'age<50':>7} {'sel/anchor':>11} {'winEedges 1st/2nd half':>22}")
    for task, rep, rd in v99_runs():
        reqs, resps, order, summary = load_run(rd)
        n_eval = summary["evaluator_call_count"] or 1
        parent_of, intent_of, born_eval = {}, {}, {}
        best_anchor = None
        for rid, r in search_responses(resps, order):
            if r["kind"] in ("new", "duplicate") and r["child_id"] is not None:
                parent_of[r["child_id"]] = r["anchor_id"]
                intent_of[r["child_id"]] = r["intent"]
                born_eval[r["child_id"]] = r["eval_count"]
            if r.get("is_new_best") and r["kind"] in ("new", "ancestral_return"):
                best_anchor = r["child_id"] if r["child_id"] is not None else r["anchor_id"]
        lineage, a = set(), best_anchor
        while a is not None and a not in lineage:
            lineage.add(a)
            a = parent_of.get(a)
        e_edges = [born_eval[x] for x in lineage
                   if intent_of.get(x) == "explore" and x in born_eval]
        first_half = sum(1 for t in e_edges if t <= n_eval / 2)
        n_sel = lineage_sel = young10 = young50 = 0
        sel_counts = Counter()
        for rid, r in search_responses(resps, order):
            q = reqs.get(rid)
            if q is None:
                continue
            a_sel = q["anchor_id"]
            n_sel += 1
            sel_counts[a_sel] += 1
            if a_sel in lineage:
                lineage_sel += 1
            age = r["eval_count"] - born_eval.get(a_sel, 0) if a_sel in born_eval else r["eval_count"]
            young10 += age <= 10
            young50 += age <= 50
        r_edges = sum(1 for x in lineage if intent_of.get(x) == "refine")
        print(f"{task}/{rep:<7} {len(lineage):7} {len(e_edges):8} {r_edges:8} "
              f"{lineage_sel / max(1, n_sel) * 100:10.1f}% {young10 / n_sel * 100:6.1f}% "
              f"{young50 / n_sel * 100:6.1f}% "
              f"{sum(sel_counts.values()) / max(1, len(sel_counts)):11.2f} "
              f"{first_half:10} | {len(e_edges) - first_half:6}")


def v97_lineage():
    print()
    print("=" * 100)
    print("F. V9.7 COMPARISON (batch 20260814_150927): top-route share and winning path depth")
    print(f"{'task/rep':12} {'topRoute':>8} {'routeShare':>10} {'pathLen':>8} {'E/R edges':>12}")
    for task, name in V99_TASKS:
        for rep in (1, 2, 3):
            g = glob.glob(f"{BASE}/{name}/traceaad_v9_7/v9_7_20260814_150927_*_rep{rep}")
            if not g:
                continue
            base = g[0]
            dec = [json.loads(l) for l in open(f"{base}/artifacts/decisions.jsonl")]
            route_sel = Counter(e["selected_root_state_id"] for e in dec if e["event"] == "route_selected")
            n = sum(route_sel.values())
            top, cnt = route_sel.most_common(1)[0]
            parent_of, intent_of = {}, {}
            cands = [json.loads(l) for l in open(f"{base}/artifacts/candidates.jsonl")]
            for c in cands:
                if c.get("kind") in ("new", "duplicate") and c.get("child_id") is not None:
                    parent_of[c["child_id"]] = c["anchor_id"]
                    intent_of[c["child_id"]] = c["intent"]
            summary = json.load(open(f"{base}/logs/summary.json"))
            best_anchor = None
            for c in cands:
                if c.get("program_id") == summary["best_program_id"] and c.get("child_id") is not None:
                    best_anchor = c["child_id"]
            lineage, a = set(), best_anchor
            while a is not None and a not in lineage:
                lineage.add(a)
                a = parent_of.get(a)
            e = sum(1 for x in lineage if intent_of.get(x) == "explore")
            r = sum(1 for x in lineage if intent_of.get(x) == "refine")
            print(f"V9.7-{task}{rep:<5} {top:8} {cnt / n * 100:9.1f}% {len(lineage):8} {e:5}/{r:<6}")


def main():
    section_allocation_geometry()
    section_operators()
    section_outcomes()
    section_lifecycle()
    section_lineage()
    v97_lineage()


if __name__ == "__main__":
    main()

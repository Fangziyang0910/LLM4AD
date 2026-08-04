"""Process-level trajectory analysis for TraceAAD v4/v5/v6 runs.

Reuses the metric ideas from papers/traj_evo_search (LLM-guided evolutionary
search trajectory analysis) and computes them from existing run logs:

- breakthrough statistics: strict sample-level global best-so-far updates;
- local refinement rate (LRR) and parent-child code distance (PCD) from
  `child_accepted` edges;
- code-distance novelty per candidate (nearest prior candidate), normalized
  per run;
- per-10-sample-window kernel entropy (H_spatial / H_fitness);
- run-level and version x task aggregation tables;
- window-level OLS (concurrent + lagged) with task/version fixed effects and
  run-clustered standard errors (statsmodels, when available).

Semantic distance is proxied by code-token Jaccard distance (see
llm4ad/method/traceaad_v6/similarity.py).  This is a proxy for the paper's
task-specific semantic distance and is explicitly marked in the report.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_samples(run_dir: Path) -> list[dict]:
    """Load evaluated candidates from new V5 artifacts or legacy profiler samples."""
    candidates_path = run_dir / "artifacts" / "candidates.jsonl"
    if candidates_path.is_file():
        samples = [
            row
            for row in _load_jsonl(candidates_path)
            if row.get("status", "ok") == "ok" or row.get("score") is not None
        ]
        # Keep failed rows out of scored analysis streams; valid_scored_samples
        # already filters non-finite scores. Include all rows with scores.
        samples = [row for row in samples if row.get("score") is not None]
        samples.sort(key=lambda s: s.get("sample_order", 0))
        return samples

    samples: list[dict] = []
    samples_dir = run_dir / "logs" / "samples"
    if not samples_dir.is_dir():
        return samples
    for f in sorted(samples_dir.glob("samples_*.json")):
        if "best" in f.name:
            continue
        samples.extend(json.load(open(f, encoding="utf-8")))
    samples.sort(key=lambda s: s.get("sample_order", 0))
    return samples


def load_run_summary(run_dir: Path) -> dict:
    logs = run_dir / "logs"
    for name in ("summary.json", "run_summary.json"):
        path = logs / name
        if path.is_file():
            return json.load(open(path, encoding="utf-8"))
    raise FileNotFoundError(f"no summary under {logs}")


def load_edge_events(run_dir: Path) -> list[dict]:
    """Parent-child edges: new artifacts/edges.jsonl or legacy child_accepted."""
    edges_path = run_dir / "artifacts" / "edges.jsonl"
    if edges_path.is_file():
        return _load_jsonl(edges_path)
    events = _load_jsonl(run_dir / "logs" / "method_events.jsonl")
    return [ev for ev in events if ev.get("event") == "child_accepted"]


def load_decision_events(run_dir: Path) -> list[dict]:
    decisions_path = run_dir / "artifacts" / "decisions.jsonl"
    if decisions_path.is_file():
        return _load_jsonl(decisions_path)
    return _load_jsonl(run_dir / "logs" / "method_events.jsonl")


def resolve_nodes_from_artifacts(
    samples: list[dict], edges: list[dict], decisions: list[dict]
) -> dict[int, Node]:
    samples_by_order = {
        int(s["sample_order"]): s
        for s in samples
        if s.get("sample_order") is not None and s.get("score") is not None
    }
    nodes: dict[int, Node] = {}
    for ev in decisions:
        if ev.get("event") != "trajectory_created":
            continue
        sample = samples_by_order.get(ev.get("sample_order"))
        if sample is None:
            continue
        nodes[int(ev["node_id"])] = Node(
            node_id=int(ev["node_id"]),
            sample_order=int(sample["sample_order"]),
            score=float(sample["score"]),
            code=sample.get("program", ""),
            operator="init",
            is_init=True,
        )
    for ev in edges:
        sample = samples_by_order.get(ev.get("sample_order"))
        if sample is None:
            continue
        child_id = int(ev["child_id"])
        nodes[child_id] = Node(
            node_id=child_id,
            sample_order=int(sample["sample_order"]),
            score=float(sample["score"]),
            code=sample.get("program", ""),
            operator=ev.get("operator", ""),
            is_init=False,
        )
    return nodes


def valid_scored_samples(samples: list[dict]) -> list[dict]:
    """Return finite, scored samples in evaluation order.

    Every process analysis uses this same candidate stream.  In particular,
    TraceAAD ``best_updated`` events are only an audit trail: a breakthrough
    is defined from the evaluated sample scores, so an event-side tie-break
    can never become a fitness breakthrough.
    """
    valid: list[dict] = []
    for index, sample in enumerate(samples):
        try:
            order = int(sample["sample_order"])
            score = float(sample["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(score):
            continue
        # Keep a private stable key without mutating the on-disk sample.
        valid.append(
            {"_analysis_index": index, **sample, "sample_order": order, "score": score}
        )
    valid.sort(key=lambda sample: (sample["sample_order"], sample["_analysis_index"]))
    return valid


def sample_level_best_stats(samples: list[dict], *, window_size: int = 10) -> dict:
    """Compute the canonical sample-level global best-so-far statistics.

    Scores in LLM4AD are normalized so that larger is better.  The first
    valid sample establishes the incumbent; only later *strict* score
    increases are fitness breakthroughs.  Equal-score shorter programs are a
    separate complexity tie-break in TraceAAD and are deliberately excluded.
    """
    scored = valid_scored_samples(samples)
    if not scored:
        return {
            "scored_samples": [],
            "breakthrough_orders": [],
            "breakthrough_windows": set(),
            "n_windows": 0,
            "breakthrough_rate_w10": None,
            "first_breakthrough_window": None,
            "last_breakthrough_window": None,
        }

    incumbent = float("-inf")
    breakthrough_orders: list[int] = []
    for sample in scored:
        score = float(sample["score"])
        if score > incumbent:
            if incumbent != float("-inf"):
                breakthrough_orders.append(int(sample["sample_order"]))
            incumbent = score

    max_order = max(int(sample["sample_order"]) for sample in scored)
    n_windows = (max_order - 1) // window_size + 1
    breakthrough_windows = {
        (order - 1) // window_size + 1 for order in breakthrough_orders
    }
    return {
        "scored_samples": scored,
        "breakthrough_orders": breakthrough_orders,
        "breakthrough_windows": breakthrough_windows,
        "n_windows": n_windows,
        "breakthrough_rate_w10": len(breakthrough_windows) / n_windows,
        "first_breakthrough_window": (
            min(breakthrough_windows) if breakthrough_windows else None
        ),
        "last_breakthrough_window": (
            max(breakthrough_windows) if breakthrough_windows else None
        ),
    }


def best_update_event_stats(events: list[dict]) -> dict:
    """Audit TraceAAD best-update events without defining breakthroughs.

    ``strict_fitness`` and ``tie_shorter`` are implementation-level reasons
    for replacing the TraceAAD incumbent.  The latter records complexity
    selection at equal fitness and is reported separately.  Older v4 logs do
    not include ``update_reason`` and therefore appear as unclassified.
    """
    updates = [
        event
        for event in events
        if event.get("event") == "best_updated"
        and event.get("previous_best_node_id") is not None
    ]
    reasons = {"strict_fitness", "tie_shorter"}
    return {
        "best_update_event_count": len(updates),
        "strict_fitness_event_count": sum(
            event.get("update_reason") == "strict_fitness" for event in updates
        ),
        "tie_shorter_count": sum(
            event.get("update_reason") == "tie_shorter" for event in updates
        ),
        "unclassified_best_update_count": sum(
            event.get("update_reason") not in reasons for event in updates
        ),
    }


def task_zscore(frame, column: str, *, task_column: str = "task"):
    """Return a task-local z-score, with zero for singleton/constant groups."""
    grouped = frame.groupby(task_column)[column]
    means = grouped.transform("mean")
    stds = grouped.transform("std").replace(0, np.nan)
    return ((frame[column] - means) / stds).fillna(0.0)


def sample_level_best_curve(samples: list[dict]) -> list[float]:
    """Return a sample-order best-so-far curve using the canonical stream."""
    scored = valid_scored_samples(samples)
    if not scored:
        return []
    by_order: dict[int, list[float]] = {}
    for sample in scored:
        by_order.setdefault(int(sample["sample_order"]), []).append(
            float(sample["score"])
        )
    curve: list[float] = []
    incumbent = float("-inf")
    for order in range(1, max(by_order) + 1):
        for score in by_order.get(order, []):
            incumbent = max(incumbent, score)
        curve.append(incumbent if incumbent != float("-inf") else 0.0)
    return curve


@dataclass
class Node:
    node_id: int
    sample_order: int
    score: float
    code: str
    operator: str
    is_init: bool


@dataclass
class Edge:
    edge_id: int
    child_id: int
    parent_id: int
    operator: str
    iteration: int | None
    outcome: str | None
    delta: float | None
    code_distance: float | None
    code_change_ratio: float | None
    sample_order: int


def resolve_nodes(
    events: list[dict], samples: list[dict], version: str
) -> dict[int, Node]:
    """Map node_id -> Node (score, code, sample_order)."""
    samples_by_order = {s["sample_order"]: s for s in samples}

    if version == "version4":
        # Init nodes: the k-th successful init sample (by sample_order)
        # corresponds to node_id k-1.
        init_samples = [
            s
            for s in samples
            if s.get("operator") == "init" and s.get("score") is not None
        ]
        nodes: dict[int, Node] = {}
        for k, s in enumerate(init_samples):
            nodes[k] = Node(
                node_id=k,
                sample_order=s["sample_order"],
                score=float(s["score"]),
                code=s.get("program", ""),
                operator="init",
                is_init=True,
            )
        # Child nodes: profiler_sample_order in child_accepted matches
        # sample_order (verified for v4 logs).  Validate against sample
        # operator/score and fall back to (operator, score) matching.
        used_orders = {n.sample_order for n in nodes.values()}
        for ev in events:
            if ev.get("event") != "child_accepted":
                continue
            child_id = ev["child_id"]
            so = ev.get("profiler_sample_order")
            cand = samples_by_order.get(so) if so is not None else None
            if (
                cand is not None
                and so not in used_orders
                and cand.get("operator") == ev.get("operator")
                and abs(
                    float(cand.get("score", math.nan))
                    - float(ev.get("score", math.nan))
                )
                < 1e-9
            ):
                sample = cand
            else:
                sample = next(
                    (
                        s
                        for s in samples
                        if s["sample_order"] not in used_orders
                        and s.get("operator") == ev.get("operator")
                        and s.get("score") is not None
                        and abs(float(s["score"]) - float(ev.get("score", math.nan)))
                        < 1e-9
                    ),
                    None,
                )
            if sample is None:
                continue
            used_orders.add(sample["sample_order"])
            nodes[child_id] = Node(
                node_id=child_id,
                sample_order=sample["sample_order"],
                score=float(sample["score"]),
                code=sample.get("program", ""),
                operator=ev.get("operator", ""),
                is_init=False,
            )
        return nodes

    # v5/v6: code_hash -> sample_order
    from llm4ad.method.traceaad_v6.complexity import code_hash

    samples_by_hash: dict[str, dict] = {}
    for s in samples:
        if s.get("program"):
            samples_by_hash.setdefault(code_hash(s["program"]), s)

    nodes = {}
    for ev in events:
        if ev.get("event") == "trajectory_created" and ev.get("stage") == "init":
            sample = samples_by_hash.get(ev.get("code_hash"))
            if sample is None:
                continue
            nodes[ev["node_id"]] = Node(
                node_id=ev["node_id"],
                sample_order=sample["sample_order"],
                score=float(sample["score"]),
                code=sample.get("program", ""),
                operator="init",
                is_init=True,
            )
        elif ev.get("event") == "child_accepted":
            sample = samples_by_hash.get(ev.get("code_hash"))
            if sample is None:
                continue
            nodes[ev["child_id"]] = Node(
                node_id=ev["child_id"],
                sample_order=sample["sample_order"],
                score=float(sample["score"]),
                code=sample.get("program", ""),
                operator=ev.get("operator", ""),
                is_init=False,
            )
    return nodes


_TOKEN_CACHE: dict[str, frozenset[str]] = {}


def _tokens(code: str) -> frozenset[str]:
    from llm4ad.method.traceaad_v6.similarity import code_tokens

    cached = _TOKEN_CACHE.get(code)
    if cached is None:
        cached = code_tokens(code)
        _TOKEN_CACHE[code] = cached
    return cached


def code_distance(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 0.0
    if not ta or not tb:
        return 1.0
    return 1.0 - len(ta & tb) / len(ta | tb)


def kernel_entropy(dists: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Gaussian-kernel entropy over a within-window distance matrix."""
    n = dists.shape[0]
    if n < 2:
        return 0.0
    off = dists[np.triu_indices(n, 1)]
    sigma = float(np.mean(off)) if off.size and np.mean(off) > 0 else 1e-6
    k = np.exp(-(dists**2) / (2.0 * sigma**2))
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    g = k @ w
    total = float(g.sum())
    if total <= 0:
        return 0.0
    q = g / total
    q = q[q > 0]
    return float(-(q * np.log(q)).sum())


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------


def analyze_run(run_dir: Path) -> tuple[dict, list[dict], list[dict]]:
    summary = load_run_summary(run_dir)
    config = {}
    cfg_path = run_dir / "run_config.json"
    if cfg_path.exists():
        config = json.load(open(cfg_path, encoding="utf-8"))

    version = str(config.get("experiment_version") or run_dir.parent.name)
    task = str(config.get("task") or run_dir.parents[2].name)
    samples = load_samples(run_dir)
    children = load_edge_events(run_dir)
    decisions = load_decision_events(run_dir)
    if (run_dir / "artifacts" / "edges.jsonl").is_file():
        nodes = resolve_nodes_from_artifacts(samples, children, decisions)
    else:
        nodes = resolve_nodes(decisions, samples, version)

    best_updates = [ev for ev in decisions if ev.get("event") == "best_updated"]
    traj_sel = [
        ev
        for ev in decisions
        if ev.get("event") in {"trajectory_selection", "trajectory_selected"}
    ]
    pop_mgmt = [
        ev
        for ev in decisions
        if ev.get("event") in {"population_management", "population_managed"}
    ]

    # Node order by sample_order for novelty computation.
    ordered = sorted(nodes.values(), key=lambda n: n.sample_order)
    scores = np.array([n.score for n in ordered], dtype=float)
    fmin, fmax = float(scores.min()), float(scores.max())
    norm_fitness = (
        (scores - fmin) / (fmax - fmin + 1e-12)
        if fmax > fmin
        else np.zeros_like(scores)
    )

    # Novelty: min code distance to any prior node.
    raw_novelty: dict[int, float] = {}
    token_sets = [_tokens(n.code) for n in ordered]

    def fast_distance(i: int, j: int) -> float:
        ta, tb = token_sets[i], token_sets[j]
        if not ta and not tb:
            return 0.0
        if not ta or not tb:
            return 1.0
        return 1.0 - len(ta & tb) / len(ta | tb)

    for i, node in enumerate(ordered):
        best_d = 1.0
        for j in range(i):
            d = fast_distance(i, j)
            if d < best_d:
                best_d = d
        raw_novelty[node.node_id] = best_d
    rmin, rmax = min(raw_novelty.values()), max(raw_novelty.values())
    norm_novelty = {
        nid: (v - rmin) / (rmax - rmin + 1e-12) if rmax > rmin else 0.0
        for nid, v in raw_novelty.items()
    }

    # Edges.
    edge_rows: list[dict] = []
    deltas: list[float] = []
    pcds: list[float] = []
    improves = 0
    outcome_counts: dict[str, int] = {}
    op_counts: dict[str, int] = {}
    change_ratios: list[float] = []
    for ev in children:
        child = nodes.get(ev.get("child_id"))
        parent = nodes.get(ev.get("parent_id"))
        if (
            child is None
            or parent is None
            or child.score is None
            or parent.score is None
        ):
            continue
        delta = float(child.score) - float(parent.score)
        d = code_distance(child.code, parent.code)
        deltas.append(delta)
        pcds.append(d)
        improves += int(delta > 0)
        outcome = ev.get("outcome")
        if outcome is None:
            outcome = "improve" if delta > 0 else "non_improve"
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        op = ev.get("operator") or "?"
        op_counts[op] = op_counts.get(op, 0) + 1
        cr = ev.get("code_change_ratio")
        if cr is not None:
            change_ratios.append(float(cr))
        edge_rows.append(
            {
                "run": run_dir.name,
                "task": task,
                "version": version,
                "edge_id": ev.get("edge_id"),
                "child_id": ev.get("child_id"),
                "parent_id": ev.get("parent_id"),
                "operator": op,
                "iteration": ev.get("iteration"),
                "outcome": outcome,
                "child_score": child.score,
                "parent_score": parent.score,
                "delta": delta,
                "improvement": bool(delta > 0),
                "code_distance": d,
                "code_change_ratio": cr,
                "child_sample_order": child.sample_order,
            }
        )

    # Canonical fitness breakthroughs come from all evaluated samples, not
    # from TraceAAD's incumbent replacement events.  The latter also include
    # equal-fitness ``tie_shorter`` complexity updates.
    best_stats = sample_level_best_stats(samples)
    event_stats = best_update_event_stats(best_updates)
    breakthrough_orders = best_stats["breakthrough_orders"]
    breakthrough_windows = best_stats["breakthrough_windows"]

    # Windows (10 samples each).
    node_by_order = {n.sample_order: n for n in ordered}
    window_rows: list[dict] = []
    n_windows = best_stats["n_windows"]
    for w in range(1, n_windows + 1):
        lo, hi = (w - 1) * 10 + 1, w * 10
        wnodes = [node_by_order[so] for so in range(lo, hi + 1) if so in node_by_order]
        if not wnodes:
            continue
        nids = [n.node_id for n in wnodes]
        mean_nov = float(np.mean([norm_novelty[nid] for nid in nids]))
        dists = np.array(
            [[code_distance(a.code, b.code) for b in wnodes] for a in wnodes],
            dtype=float,
        )
        fit_weights = np.array(
            [norm_fitness[ordered.index(n)] for n in wnodes], dtype=float
        )
        h_spatial = kernel_entropy(dists, None)
        h_fitness = kernel_entropy(dists, fit_weights)
        window_rows.append(
            {
                "run": run_dir.name,
                "task": task,
                "version": version,
                "window": w,
                "n_nodes": len(wnodes),
                "mean_novelty": mean_nov,
                "h_spatial": h_spatial,
                "h_fitness": h_fitness,
                "breakthrough": int(w in breakthrough_windows),
            }
        )

    n_edges = len(edge_rows)
    n_nodes = len(nodes)
    run_metrics = {
        "run": run_dir.name,
        "task": task,
        "version": version,
        "model": config.get("llm", {}).get("model"),
        "temperature": config.get("llm", {}).get("temperature"),
        "repeat": config.get("repeat"),
        "num_samples": summary.get("num_samples"),
        "llm_calls": summary.get("llm_call_count"),
        "best_score": summary.get("best_score"),
        "best_sample_order": summary.get("best_sample_order"),
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        # Canonical metric: strict global best-so-far score increases.
        "breakthrough_count": len(breakthrough_orders),
        "breakthrough_rate_w10": best_stats["breakthrough_rate_w10"],
        "first_breakthrough_window": best_stats["first_breakthrough_window"],
        "last_breakthrough_window": best_stats["last_breakthrough_window"],
        "n_windows": n_windows,
        # Event-side audit metrics.  ``tie_shorter`` is complexity progress,
        # never a fitness breakthrough.
        **event_stats,
        "lrr": improves / n_edges if n_edges else None,
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "mean_pcd": float(np.mean(pcds)) if pcds else None,
        "mean_code_change_ratio": float(np.mean(change_ratios))
        if change_ratios
        else None,
        "avg_novelty": float(np.mean(list(norm_novelty.values()))),
        "init_novelty": float(
            np.mean([norm_novelty[n.node_id] for n in ordered if n.is_init])
        ),
        "mean_h_spatial": float(np.mean([r["h_spatial"] for r in window_rows])),
        "mean_h_fitness": float(np.mean([r["h_fitness"] for r in window_rows])),
        "operator_mix": json.dumps(op_counts, ensure_ascii=False),
        "outcome_mix": json.dumps(outcome_counts, ensure_ascii=False),
        "trajectory_selection_count": len(traj_sel),
        "population_management_count": len(pop_mgmt),
        "mean_top5_mass": (
            float(
                np.mean(
                    [
                        t["top5_probability_mass"]
                        for t in traj_sel
                        if t.get("top5_probability_mass") is not None
                    ]
                )
            )
            if any(t.get("top5_probability_mass") is not None for t in traj_sel)
            else None
        ),
        "mean_effective_candidates": (
            float(
                np.mean(
                    [
                        t["effective_candidate_count"]
                        for t in traj_sel
                        if t.get("effective_candidate_count") is not None
                    ]
                )
            )
            if any(t.get("effective_candidate_count") is not None for t in traj_sel)
            else None
        ),
    }
    return run_metrics, window_rows, edge_rows


# ---------------------------------------------------------------------------
# Aggregation + regression
# ---------------------------------------------------------------------------


def aggregate(metrics: list[dict]) -> dict:
    import pandas as pd

    df = pd.DataFrame(metrics)
    cols = [
        "best_score",
        "breakthrough_count",
        "breakthrough_rate_w10",
        "strict_fitness_event_count",
        "tie_shorter_count",
        "unclassified_best_update_count",
        "lrr",
        "mean_delta",
        "mean_pcd",
        "mean_code_change_ratio",
        "avg_novelty",
        "init_novelty",
        "mean_h_spatial",
        "mean_h_fitness",
        "llm_calls",
    ]
    out: dict[str, dict] = {}
    for (version, task), g in df.groupby(["version", "task"]):
        key = f"{version}|{task}"
        out[key] = {
            c: {
                "mean": round(float(g[c].mean()), 4) if g[c].notna().any() else None,
                "std": round(float(g[c].std()), 4) if g[c].notna().sum() > 1 else None,
            }
            for c in cols
        }
        out[key]["n_reps"] = len(g)
    version_main: dict[str, dict] = {}
    for version, g in df.groupby("version"):
        version_main[version] = {
            c: {
                "mean": round(float(g[c].mean()), 4) if g[c].notna().any() else None,
                "std": round(float(g[c].std()), 4) if g[c].notna().sum() > 1 else None,
            }
            for c in cols
        }
        version_main[version]["n_runs"] = len(g)
    return {"by_version_task": out, "by_version": version_main}


def run_regressions(window_rows: list[dict], run_metrics: list[dict]) -> dict:
    try:
        import statsmodels.api as sm
        import pandas as pd
    except Exception as exc:  # pragma: no cover
        return {"error": f"statsmodels unavailable: {exc}"}

    wdf = pd.DataFrame(window_rows)
    for c in ["mean_novelty", "h_spatial", "h_fitness", "window"]:
        z = (wdf[c] - wdf[c].mean()) / (wdf[c].std() + 1e-12)
        wdf[f"{c}_z"] = z
    wdf["nov_x_spatial_z"] = wdf["mean_novelty_z"] * wdf["h_spatial_z"]
    wdf["y_lag"] = wdf.groupby("run")["breakthrough"].shift(-1)

    fe_cols = ["task", "version"]

    def fit(df, ycol, cluster):
        y = df[ycol].astype(float)
        x = df[
            [
                "mean_novelty_z",
                "h_spatial_z",
                "h_fitness_z",
                "window_z",
                "nov_x_spatial_z",
                *fe_cols,
            ]
        ].copy()
        x = pd.get_dummies(x, columns=fe_cols, drop_first=True, dtype=float)
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})
        return {
            "n": int(model.nobs),
            "r2": float(model.rsquared),
            "params": {
                k: {
                    "coef": float(v),
                    "pvalue": float(model.pvalues[k]),
                    "se": float(model.bse[k]),
                }
                for k, v in model.params.items()
            },
        }

    concurrent = fit(wdf.dropna(subset=["breakthrough"]), "breakthrough", "run")
    lagged_df = wdf.dropna(subset=["y_lag"])
    lagged = fit(lagged_df, "y_lag", "run")

    # Run-level regression: normalize the outcome and descriptors within task
    # before comparing unlike objective scales (TSP/CVRP/OBP/OP).
    rdf = pd.DataFrame(run_metrics)
    for c in ["lrr", "mean_pcd", "breakthrough_rate_w10", "avg_novelty", "best_score"]:
        rdf[f"{c}_z"] = task_zscore(rdf, c)
    y = rdf["best_score_z"]
    x = rdf[
        [
            "lrr_z",
            "mean_pcd_z",
            "breakthrough_rate_w10_z",
            "avg_novelty_z",
            "task",
            "version",
        ]
    ].copy()
    x = pd.get_dummies(x, columns=["task", "version"], drop_first=True, dtype=float)
    x = sm.add_constant(x)
    run_model = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": rdf["run"]})
    run_level = {
        "n": int(run_model.nobs),
        "r2": float(run_model.rsquared),
        "params": {
            k: {
                "coef": float(v),
                "pvalue": float(run_model.pvalues[k]),
                "se": float(run_model.bse[k]),
            }
            for k, v in run_model.params.items()
        },
    }
    return {
        "concurrent": concurrent,
        "lagged": lagged,
        "run_level": run_level,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def make_figures(
    window_rows: list[dict], run_metrics: list[dict], out_dir: Path
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    wdf = pd.DataFrame(window_rows)
    rdf = pd.DataFrame(run_metrics)
    tasks = sorted(rdf["task"].unique())
    versions = ["version4", "version5", "version6"]
    colors = {"version4": "#F4A261", "version5": "#009E73", "version6": "#7B68EE"}
    dir_suffix = {"version4": "v4", "version5": "v5", "version6": "v6"}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, task in zip(axes.flat, tasks):
        tdf = rdf[rdf["task"] == task]
        vmin, vmax = float(tdf["best_score"].min()), float(tdf["best_score"].max())
        for ver in versions:
            vdf = tdf[tdf["version"] == ver]
            curves = []
            for _, row in vdf.iterrows():
                # find run dir by name under experiments
                cand = list(
                    ROOT.glob(
                        f"experiments/{task}/traceaad_{dir_suffix[ver]}/**/"
                        f"{row['run']}/logs"
                    )
                )
                if not cand:
                    continue
                best = best_sofar_curve(cand[0])
                curves.append(best)
            if not curves:
                continue
            lens = min(len(c) for c in curves)
            arr = np.array([c[:lens] for c in curves])
            norm = (arr - vmin) / (vmax - vmin + 1e-12)
            mean, std = norm.mean(axis=0), norm.std(axis=0)
            x = np.arange(1, lens + 1)
            ax.plot(x, mean, label=ver, color=colors[ver])
            ax.fill_between(x, mean - std, mean + std, alpha=0.18, color=colors[ver])
        ax.set_title(task)
        ax.set_xlabel("sample order")
        ax.set_ylabel("normalized best-so-far")
    fig.suptitle("TraceAAD v4/v5/v6 best-so-far curves (mean ± std over reps)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p1 = out_dir / "fig_best_sofar.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, task in zip(axes.flat, tasks):
        tdf = wdf[wdf["task"] == task]
        for ver in versions:
            vdf = tdf[tdf["version"] == ver].groupby("window")["h_spatial"].mean()
            if vdf.empty:
                continue
            ax.plot(vdf.index, vdf.values, label=ver, color=colors[ver])
        ax.set_title(f"{task} — mean H_spatial by window")
        ax.set_xlabel("window (10 samples)")
        ax.set_ylabel("H_spatial")
    fig.suptitle("TraceAAD v4/v5/v6 search localization dynamics")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p2 = out_dir / "fig_h_spatial.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return [p1, p2]


def best_sofar_curve(logs_dir: Path) -> list[float]:
    run_dir = logs_dir.parent
    sample_curve = sample_level_best_curve(load_samples(run_dir))
    if sample_curve:
        return sample_curve

    # Legacy fallback for methods/runs that have no profiler sample files.
    events_path = logs_dir / "method_events.jsonl"
    if not events_path.is_file():
        return []
    by_order = {}
    for line in open(events_path, encoding="utf-8"):
        ev = json.loads(line)
        if ev.get("event") == "program_evaluated" and ev.get("status") == "ok":
            by_order[int(ev["sample_order"])] = float(ev["score"])
    if not by_order:
        return []
    cur = float("-inf")
    curve = []
    for so in range(1, max(by_order) + 1):
        if so in by_order:
            cur = max(cur, by_order[so])
        curve.append(cur if cur != float("-inf") else 0.0)
    return curve


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=ROOT / "docs" / "research" / "traceaad_v4-v6_analysis"
    )
    parser.add_argument("--no-regress", action="store_true")
    parser.add_argument(
        "--run-prefix",
        action="append",
        default=[],
        help=(
            "Only include run directories whose names start with this prefix; "
            "repeat the option to select multiple batches."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = []
    summary_paths = list(
        ROOT.glob("experiments/*/traceaad_v*/version*/*/logs/run_summary.json")
    ) + list(ROOT.glob("experiments/*/traceaad_v*/version*/*/logs/summary.json"))
    seen: set[Path] = set()
    for summary in sorted(summary_paths):
        run_dir = summary.parent.parent
        if run_dir in seen:
            continue
        seen.add(run_dir)
        if "_eval_proxy" in run_dir.name:
            continue
        if args.run_prefix and not any(
            run_dir.name.startswith(prefix) for prefix in args.run_prefix
        ):
            continue
        try:
            st = json.load(open(summary, encoding="utf-8"))
        except Exception:
            continue
        if st.get("status") != "finished" or st.get("num_samples", 0) < 1000:
            continue
        run_dirs.append(run_dir)
    if args.run_prefix:
        print(f"run-prefix filter: {args.run_prefix}", flush=True)
    else:
        print(
            "no run-prefix filter: all matching finished runs are included; "
            "use --run-prefix to isolate a protocol batch",
            flush=True,
        )
    print(f"found {len(run_dirs)} finished runs", flush=True)

    all_metrics: list[dict] = []
    all_windows: list[dict] = []
    all_edges: list[dict] = []
    for rd in run_dirs:
        rm, wrows, erows = analyze_run(rd)
        all_metrics.append(rm)
        all_windows.extend(wrows)
        all_edges.extend(erows)
        print(
            f"  {rd.name}: nodes={rm['n_nodes']} edges={rm['n_edges']} "
            f"br={rm['breakthrough_count']} lrr={rm['lrr']} pcd={rm['mean_pcd']}",
            flush=True,
        )

    import pandas as pd

    pd.DataFrame(all_metrics).to_csv(out_dir / "run_metrics.csv", index=False)
    pd.DataFrame(all_windows).to_csv(out_dir / "window_metrics.csv", index=False)
    pd.DataFrame(all_edges).to_csv(out_dir / "edge_metrics.csv", index=False)

    agg = aggregate(all_metrics)
    with open(out_dir / "aggregate.json", "w", encoding="utf-8") as fh:
        json.dump(agg, fh, ensure_ascii=False, indent=2)

    regressions = {}
    if not args.no_regress:
        regressions = run_regressions(all_windows, all_metrics)
        with open(out_dir / "regressions.json", "w", encoding="utf-8") as fh:
            json.dump(regressions, fh, ensure_ascii=False, indent=2)

    figs = make_figures(all_windows, all_metrics, out_dir)
    print("figures:", [f.name for f in figs], flush=True)
    print("outputs written to", out_dir, flush=True)


if __name__ == "__main__":
    main()

"""Cross-method process analysis for TraceAAD v4-v6 and five baselines.

Methods: traceaad v4/v5/v6, mcts_ahd, eoh, reevo, shinka_evo, pathwise.

Reuses the metric definitions of the traj_evo_search paper:
- best-so-far / breakthrough statistics (sample-based, consistent across
  methods: a sample is a breakthrough when its score strictly beats all
  earlier samples);
- local refinement rate (LRR) and parent-child code distance (PCD), where the
  method logs allow parent-child reconstruction;
- code-distance novelty (nearest prior candidate), normalized per run;
- per-10-sample-window kernel entropy (H_spatial / H_fitness);
- window-level OLS (concurrent + lagged) with task/method fixed effects and
  run-clustered standard errors;
- run-level within-task correlations and method aggregation.

Per-method adapters:
- traceaad v4/v5/v6: `child_accepted` edges (code_hash / profiler order);
- mcts_ahd: `expand` events, parent sample matched by parent_score;
- shinka_evo: `archive_update` (program_id -> sample_order) +
  `patch_attempt` (parent_id -> child sample);
- pathwise: `population/pop_*.json` (node_id -> function) +
  `pathwise/entailment_steps.jsonl` (parent/child node ids);
- eoh / reevo: no parent-child logs -> LRR/PCD unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:  # Direct script execution and package/test imports use different paths.
    from analyze_traceaad_process import (
        _load_jsonl,
        best_update_event_stats,
        code_distance,
        kernel_entropy,
        load_samples,
        sample_level_best_curve,
        sample_level_best_stats,
        task_zscore,
        valid_scored_samples,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised by package import
    from experiments.analysis.analyze_traceaad_process import (
        _load_jsonl,
        best_update_event_stats,
        code_distance,
        kernel_entropy,
        load_samples,
        sample_level_best_curve,
        sample_level_best_stats,
        task_zscore,
        valid_scored_samples,
    )

ROOT = Path(__file__).resolve().parents[2]
MIN_SAMPLES = 900  # excludes smoke runs and the old 500-sample PathWise protocol


def discover_runs(run_prefixes: list[str] | None = None) -> list[dict]:
    run_prefixes = run_prefixes or []

    def selected(run_dir: Path) -> bool:
        return not run_prefixes or any(
            run_dir.name.startswith(prefix) for prefix in run_prefixes
        )

    runs: list[dict] = []
    # TraceAAD versions (legacy run_summary.json and new summary.json).
    summary_paths = list(
        ROOT.glob("experiments/*/traceaad_v*/version*/*/logs/run_summary.json")
    ) + list(ROOT.glob("experiments/*/traceaad_v*/version*/*/logs/summary.json"))
    seen_runs: set[Path] = set()
    for summary in sorted(summary_paths):
        run_dir = summary.parent.parent
        if run_dir in seen_runs:
            continue
        seen_runs.add(run_dir)
        if "_eval_proxy" in run_dir.name:
            continue
        if not selected(run_dir):
            continue
        cfg_path = run_dir / "run_config.json"
        cfg = json.load(open(cfg_path, encoding="utf-8")) if cfg_path.exists() else {}
        version = str(cfg.get("experiment_version") or run_dir.parent.name)
        task = str(cfg.get("task") or run_dir.parents[2].name)
        runs.append(
            {
                "method": version,
                "task": task,
                "run_dir": run_dir,
                "run_name": run_dir.name,
            }
        )
    # Other methods.
    for method in ("mcts_ahd", "eoh", "reevo", "shinka_evo", "pathwise"):
        for summary in sorted(
            ROOT.glob(f"experiments/*/{method}/*/logs/run_summary.json")
        ):
            run_dir = summary.parent.parent
            if "smoke" in run_dir.name or "_eval_proxy" in run_dir.name:
                continue
            if not selected(run_dir):
                continue
            task = summary.parents[3].name
            runs.append(
                {
                    "method": method,
                    "task": task,
                    "run_dir": run_dir,
                    "run_name": run_dir.name,
                }
            )
    return runs


def _latest_prior_sample_with_score(
    samples_by_order: dict[int, dict], score: float, before: int
) -> int | None:
    best = None
    for so, s in samples_by_order.items():
        if so < before and s.get("score") is not None:
            if abs(float(s["score"]) - score) < 1e-9:
                if best is None or so > best:
                    best = so
    return best


def extract_edges(method: str, run_dir: Path, samples: list[dict]) -> list[dict]:
    """Return edges with child/parent sample_order (empty when unavailable)."""
    logs = run_dir / "logs"
    samples_by_order = {s["sample_order"]: s for s in samples}
    edges: list[dict] = []

    if method.startswith("version"):
        # TraceAAD: prefer new artifacts layout; fall back to legacy method_events.
        try:
            from analyze_traceaad_process import (
                load_decision_events,
                load_edge_events,
                resolve_nodes,
                resolve_nodes_from_artifacts,
            )
        except ModuleNotFoundError:  # pragma: no cover - package import path
            from experiments.analysis.analyze_traceaad_process import (
                load_decision_events,
                load_edge_events,
                resolve_nodes,
                resolve_nodes_from_artifacts,
            )

        edge_events = load_edge_events(run_dir)
        if (run_dir / "artifacts" / "edges.jsonl").is_file():
            decisions = load_decision_events(run_dir)
            nodes = resolve_nodes_from_artifacts(samples, edge_events, decisions)
            for ev in edge_events:
                child = nodes.get(ev.get("child_id"))
                parent = nodes.get(ev.get("parent_id"))
                if child is None or parent is None:
                    continue
                edges.append(
                    {
                        "child_sample_order": child.sample_order,
                        "parent_sample_order": parent.sample_order,
                        "operator": ev.get("operator") or "",
                    }
                )
            return edges

        events = edge_events  # legacy: child_accepted list from method_events
        # When edges.jsonl missing, load_edge_events returns child_accepted rows.
        # Older resolve_nodes still needs full method_events for init nodes.
        legacy_events = _load_jsonl(logs / "method_events.jsonl")
        nodes = resolve_nodes(legacy_events or events, samples, method)
        for ev in events:
            if ev.get("event") not in {None, "child_accepted"} and "parent_id" not in ev:
                continue
            if ev.get("event") == "child_accepted" or (
                "parent_id" in ev and "child_id" in ev
            ):
                child = nodes.get(ev.get("child_id"))
                parent = nodes.get(ev.get("parent_id"))
                if child is None or parent is None:
                    continue
                edges.append(
                    {
                        "child_sample_order": child.sample_order,
                        "parent_sample_order": parent.sample_order,
                        "operator": ev.get("operator") or "",
                    }
                )
        return edges

    if method == "mcts_ahd":
        events_path = logs / "mcts_events.jsonl"
        events = _load_jsonl(events_path) if events_path.exists() else []
        for ev in events:
            if ev.get("event") != "expand":
                continue
            child_so = ev.get("sample_order")
            parent_score = ev.get("parent_score")
            if child_so is None or parent_score is None or ev.get("root_parent"):
                continue
            parent_so = _latest_prior_sample_with_score(
                samples_by_order, float(parent_score), int(child_so)
            )
            if parent_so is None:
                continue
            edges.append(
                {
                    "child_sample_order": int(child_so),
                    "parent_sample_order": parent_so,
                    "operator": ev.get("operator") or "",
                }
            )
        return edges

    if method == "shinka_evo":
        events = _load_jsonl(logs / "method_events.jsonl")
        pid_to_sample: dict[str, int] = {}
        for ev in events:
            if ev.get("event") == "archive_update":
                so = ev.get("profiler_sample_order")
                if ev.get("program_id") and so is not None and so in samples_by_order:
                    pid_to_sample[ev["program_id"]] = int(so)
        for ev in events:
            if ev.get("event") != "patch_attempt" or not ev.get("success"):
                continue
            child_so = ev.get("profiler_sample_order")
            parent_so = pid_to_sample.get(ev.get("parent_id"))
            if child_so is None or parent_so is None:
                continue
            if child_so in samples_by_order and child_so != parent_so:
                edges.append(
                    {
                        "child_sample_order": int(child_so),
                        "parent_sample_order": parent_so,
                        "operator": ev.get("patch_type") or "patch",
                    }
                )
        return edges

    if method == "pathwise":
        # node_id -> function from population snapshots (last occurrence wins).
        node_function: dict[str, str] = {}
        pop_dir = logs / "population"
        if pop_dir.exists():
            for f in sorted(pop_dir.glob("pop_*.json")):
                try:
                    entries = json.load(open(f, encoding="utf-8"))
                except Exception:
                    continue
                for entry in entries:
                    nid = entry.get("node_id")
                    fn = entry.get("function")
                    if nid and fn:
                        node_function[nid] = fn
        # function -> sample_order from samples (exact function string).
        function_to_sample: dict[str, int] = {}
        for s in samples:
            fn = s.get("function")
            if fn:
                function_to_sample.setdefault(fn, s["sample_order"])

        def node_to_sample(nid: str) -> int | None:
            fn = node_function.get(nid)
            if fn is None:
                return None
            return function_to_sample.get(fn)

        entail_path = logs / "pathwise" / "entailment_steps.jsonl"
        if not entail_path.exists():
            return edges
        for line in open(entail_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            edge = row.get("edge") or {}
            child_nid = edge.get("child") or row.get("selected_node", {}).get("node_id")
            parents = edge.get("parents") or []
            child_so = node_to_sample(child_nid) if child_nid else None
            if child_so is None:
                continue
            for pnid in parents:
                parent_so = node_to_sample(pnid)
                if parent_so is None or parent_so == child_so:
                    continue
                edges.append(
                    {
                        "child_sample_order": child_so,
                        "parent_sample_order": parent_so,
                        "operator": "entail",
                    }
                )
        return edges

    return edges  # eoh / reevo


def analyze_run(
    method: str, task: str, run_dir: Path
) -> tuple[dict, list[dict], list[dict]]:
    logs = run_dir / "logs"
    try:
        from analyze_traceaad_process import load_run_summary
    except ModuleNotFoundError:  # pragma: no cover
        from experiments.analysis.analyze_traceaad_process import load_run_summary

    try:
        summary = load_run_summary(run_dir)
    except FileNotFoundError:
        summary = json.load(open(logs / "run_summary.json", encoding="utf-8"))
    samples = valid_scored_samples(load_samples(run_dir))
    samples_by_order = {s["sample_order"]: s for s in samples}

    edges = extract_edges(method, run_dir, samples)
    edge_rows: list[dict] = []
    deltas: list[float] = []
    pcds: list[float] = []
    improves = 0
    op_counts: dict[str, int] = {}
    for e in edges:
        child = samples_by_order.get(e["child_sample_order"])
        parent = samples_by_order.get(e["parent_sample_order"])
        if child is None or parent is None:
            continue
        delta = float(child["score"]) - float(parent["score"])
        d = code_distance(child["program"], parent["program"])
        deltas.append(delta)
        pcds.append(d)
        improves += int(delta > 0)
        op_counts[e["operator"]] = op_counts.get(e["operator"], 0) + 1
        edge_rows.append(
            {
                "run": run_dir.name,
                "task": task,
                "method": method,
                "child_sample_order": child["sample_order"],
                "parent_sample_order": parent["sample_order"],
                "operator": e["operator"],
                "child_score": child["score"],
                "parent_score": parent["score"],
                "delta": delta,
                "improvement": bool(delta > 0),
                "code_distance": d,
            }
        )

    scores = np.array([s["score"] for s in samples], dtype=float)
    fmin, fmax = float(scores.min()), float(scores.max())
    norm_fitness = (
        (scores - fmin) / (fmax - fmin + 1e-12)
        if fmax > fmin
        else np.zeros_like(scores)
    )

    # Canonical sample-level global best-so-far statistics shared with the
    # TraceAAD-specific analysis.  Equal-score shorter programs are reported
    # as a separate complexity tie-break when the method logs that event.
    best_stats = sample_level_best_stats(samples)
    breakthroughs = best_stats["breakthrough_orders"]

    # Novelty: min code distance to prior samples.
    raw_novelty: list[float] = []
    for i, s in enumerate(samples):
        best_d = 1.0
        for j in range(i):
            d = code_distance(s["program"], samples[j]["program"])
            if d < best_d:
                best_d = d
        raw_novelty.append(best_d)
    rmin, rmax = min(raw_novelty), max(raw_novelty)
    norm_novelty = [
        (v - rmin) / (rmax - rmin + 1e-12) if rmax > rmin else 0.0 for v in raw_novelty
    ]

    # Operator mix from samples (fallback: edges).
    sample_ops: dict[str, int] = {}
    for s in samples:
        op = s.get("operator")
        if op:
            sample_ops[op] = sample_ops.get(op, 0) + 1
    if not sample_ops and method in ("eoh", "reevo"):
        # sample_registered / operator_start events carry the operator.
        for ev in _load_jsonl(logs / "method_events.jsonl"):
            op = ev.get("operator")
            so = ev.get("sample_order")
            if (
                op
                and so is not None
                and ev.get("event") in ("sample_registered", "operator_start")
            ):
                sample_ops[op] = sample_ops.get(op, 0) + 1
    operator_mix = sample_ops if sample_ops else op_counts

    # Windows of 10 samples.
    window_rows: list[dict] = []
    max_win = best_stats["n_windows"]
    break_windows = best_stats["breakthrough_windows"]
    for w in range(1, max_win + 1):
        lo, hi = (w - 1) * 10 + 1, w * 10
        idx = [i for i, s in enumerate(samples) if lo <= s["sample_order"] <= hi]
        if not idx:
            continue
        mean_nov = float(np.mean([norm_novelty[i] for i in idx]))
        dists = np.array(
            [
                [
                    code_distance(samples[i]["program"], samples[j]["program"])
                    for j in idx
                ]
                for i in idx
            ],
            dtype=float,
        )
        fit_w = np.array([norm_fitness[i] for i in idx], dtype=float)
        window_rows.append(
            {
                "run": run_dir.name,
                "task": task,
                "method": method,
                "window": w,
                "n_nodes": len(idx),
                "mean_novelty": mean_nov,
                "h_spatial": kernel_entropy(dists, None),
                "h_fitness": kernel_entropy(dists, fit_w),
                "breakthrough": int(w in break_windows),
            }
        )

    cfg_path = run_dir / "run_config.json"
    cfg = json.load(open(cfg_path, encoding="utf-8")) if cfg_path.exists() else {}
    events_path = logs / "method_events.jsonl"
    event_stats = best_update_event_stats(
        _load_jsonl(events_path) if events_path.exists() else []
    )
    run_metrics = {
        "run": run_dir.name,
        "task": task,
        "method": method,
        "model": cfg.get("llm", {}).get("model"),
        "num_samples": summary.get("num_samples"),
        "llm_calls": summary.get("llm_call_count"),
        "best_score": summary.get("best_score"),
        "best_sample_order": summary.get("best_sample_order"),
        "breakthrough_count": len(breakthroughs),
        "breakthrough_rate_w10": best_stats["breakthrough_rate_w10"],
        "first_breakthrough_window": best_stats["first_breakthrough_window"],
        "last_breakthrough_window": best_stats["last_breakthrough_window"],
        "n_windows": max_win,
        **event_stats,
        "lrr": improves / len(edge_rows) if edge_rows else None,
        "mean_delta": float(np.mean(deltas)) if deltas else None,
        "mean_pcd": float(np.mean(pcds)) if pcds else None,
        "n_edges": len(edge_rows),
        "avg_novelty": float(np.mean(norm_novelty)),
        "init_novelty": float(np.mean(norm_novelty[: min(30, len(norm_novelty))])),
        "mean_h_spatial": float(np.mean([r["h_spatial"] for r in window_rows])),
        "mean_h_fitness": float(np.mean([r["h_fitness"] for r in window_rows])),
        "operator_mix": json.dumps(operator_mix, ensure_ascii=False),
    }
    return run_metrics, window_rows, edge_rows


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

    def fit(df, ycol):
        y = df[ycol].astype(float)
        x = df[
            [
                "mean_novelty_z",
                "h_spatial_z",
                "h_fitness_z",
                "window_z",
                "nov_x_spatial_z",
                "task",
                "method",
            ]
        ].copy()
        x = pd.get_dummies(x, columns=["task", "method"], drop_first=True, dtype=float)
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit(cov_type="cluster", cov_kwds={"groups": df["run"]})
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

    concurrent = fit(wdf.dropna(subset=["breakthrough"]), "breakthrough")
    lagged = fit(wdf.dropna(subset=["y_lag"]), "y_lag")

    # Run-level: within-task z best vs descriptors (LRR/PCD only where logged).
    rdf = pd.DataFrame(run_metrics)
    with_edges = rdf[rdf["n_edges"] > 0].copy()
    for c in ["lrr", "mean_pcd", "breakthrough_rate_w10", "avg_novelty"]:
        with_edges[f"{c}_z"] = task_zscore(with_edges, c)
    with_edges["best_z_task"] = task_zscore(with_edges, "best_score")
    y = with_edges["best_z_task"]
    x = with_edges[
        [
            "lrr_z",
            "mean_pcd_z",
            "breakthrough_rate_w10_z",
            "avg_novelty_z",
            "task",
            "method",
        ]
    ].copy()
    x = pd.get_dummies(x, columns=["task", "method"], drop_first=True, dtype=float)
    x = sm.add_constant(x)
    run_model = sm.OLS(y, x).fit(
        cov_type="cluster", cov_kwds={"groups": with_edges["run"]}
    )
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
    methods = sorted(rdf["method"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    colors = {m: palette[i] for i, m in enumerate(methods)}

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, task in zip(axes.flat, tasks):
        tdf = rdf[rdf["task"] == task]
        vmin, vmax = float(tdf["best_score"].min()), float(tdf["best_score"].max())
        for method in methods:
            mdf = tdf[tdf["method"] == method]
            curves = []
            for _, row in mdf.iterrows():
                logs = run_logs_dir(row["task"], row["method"], row["run"])
                if logs is None:
                    continue
                curves.append(best_sofar_curve(logs))
            if not curves:
                continue
            lens = min(len(c) for c in curves)
            arr = np.array([c[:lens] for c in curves])
            norm = (arr - vmin) / (vmax - vmin + 1e-12)
            mean = norm.mean(axis=0)
            std = norm.std(axis=0)
            x = np.arange(1, lens + 1)
            ax.plot(x, mean, label=method, color=colors[method])
            ax.fill_between(x, mean - std, mean + std, alpha=0.12, color=colors[method])
        ax.set_title(task)
        ax.set_xlabel("sample order")
        ax.set_ylabel("normalized best-so-far")
        ax.legend(fontsize=7)
    fig.suptitle("Cross-method best-so-far curves (mean ± std over reps)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p1 = out_dir / "fig_best_sofar.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, task in zip(axes.flat, tasks):
        tdf = wdf[wdf["task"] == task]
        for method in methods:
            vdf = tdf[tdf["method"] == method].groupby("window")["h_spatial"].mean()
            if vdf.empty:
                continue
            ax.plot(vdf.index, vdf.values, label=method, color=colors[method])
        ax.set_title(f"{task} — mean H_spatial by window")
        ax.set_xlabel("window (10 samples)")
        ax.set_ylabel("H_spatial")
        ax.legend(fontsize=7)
    fig.suptitle("Cross-method search localization dynamics")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p2 = out_dir / "fig_h_spatial.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return [p1, p2]


def run_logs_dir(task: str, method: str, run_name: str) -> Path | None:
    if method.startswith("version"):
        suffix = method[-1]
        cands = list(
            ROOT.glob(f"experiments/{task}/traceaad_v{suffix}/**/{run_name}/logs")
        )
        return cands[0] if cands else None
    p = ROOT / "experiments" / task / method / run_name / "logs"
    return p if p.exists() else None


def best_sofar_curve(logs_dir: Path) -> list[float]:
    sample_curve = sample_level_best_curve(load_samples(logs_dir.parent.parent))
    if sample_curve:
        return sample_curve

    # Legacy fallback for methods/runs that have no profiler sample files.
    by_order = {}
    for ev in _load_jsonl(logs_dir / "method_events.jsonl"):
        if ev.get("event") == "program_evaluated" and ev.get("status") == "ok":
            by_order[int(ev["sample_order"])] = float(ev["score"])
    if not by_order:
        # Fallback: samples files.
        for s in load_samples(logs_dir.parent):
            if s.get("score") is not None:
                by_order[int(s["sample_order"])] = float(s["score"])
    if not by_order:
        return []
    cur = float("-inf")
    curve = []
    for so in range(1, max(by_order) + 1):
        if so in by_order:
            cur = max(cur, by_order[so])
        curve.append(cur if cur != float("-inf") else 0.0)
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", default=ROOT / "docs" / "research" / "cross_method_analysis"
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

    runs = discover_runs(args.run_prefix)
    kept = []
    for r in runs:
        summary = json.load(
            open(r["run_dir"] / "logs" / "run_summary.json", encoding="utf-8")
        )
        if summary.get("status") != "finished":
            continue
        if (summary.get("num_samples") or 0) < MIN_SAMPLES:
            continue
        kept.append(r)
    if args.run_prefix:
        print(f"run-prefix filter: {args.run_prefix}", flush=True)
    else:
        print(
            "no run-prefix filter: all matching finished runs are included; "
            "use --run-prefix to isolate a protocol batch",
            flush=True,
        )
    print(f"found {len(kept)} finished runs", flush=True)

    all_metrics: list[dict] = []
    all_windows: list[dict] = []
    all_edges: list[dict] = []
    for r in kept:
        rm, wrows, erows = analyze_run(r["method"], r["task"], r["run_dir"])
        all_metrics.append(rm)
        all_windows.extend(wrows)
        all_edges.extend(erows)
        print(
            f"  {r['method']:12s} {r['task']:18s} {r['run_name']}: "
            f"br={rm['breakthrough_count']:3d} lrr={rm['lrr']} pcd={rm['mean_pcd']}",
            flush=True,
        )

    import pandas as pd

    pd.DataFrame(all_metrics).to_csv(out_dir / "run_metrics.csv", index=False)
    pd.DataFrame(all_windows).to_csv(out_dir / "window_metrics.csv", index=False)
    pd.DataFrame(all_edges).to_csv(out_dir / "edge_metrics.csv", index=False)

    agg: dict[str, dict] = {}
    rdf = pd.DataFrame(all_metrics)
    cols = [
        "best_score",
        "breakthrough_count",
        "breakthrough_rate_w10",
        "strict_fitness_event_count",
        "tie_shorter_count",
        "unclassified_best_update_count",
        "lrr",
        "mean_pcd",
        "avg_novelty",
        "init_novelty",
        "mean_h_spatial",
        "mean_h_fitness",
        "llm_calls",
    ]
    for (method, task), g in rdf.groupby(["method", "task"]):
        agg[f"{method}|{task}"] = {
            c: {
                "mean": round(float(g[c].mean()), 4) if g[c].notna().any() else None,
                "std": round(float(g[c].std()), 4) if g[c].notna().sum() > 1 else None,
            }
            for c in cols
        }
        agg[f"{method}|{task}"]["n_reps"] = len(g)
    for method, g in rdf.groupby("method"):
        agg[f"{method}|ALL"] = {
            c: {
                "mean": round(float(g[c].mean()), 4) if g[c].notna().any() else None,
                "std": round(float(g[c].std()), 4) if g[c].notna().sum() > 1 else None,
            }
            for c in cols
        }
        agg[f"{method}|ALL"]["n_runs"] = len(g)
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

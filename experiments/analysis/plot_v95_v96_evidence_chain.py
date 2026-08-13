"""Figures and aggregate tables for the V9.5 vs V9.6 evidence-chain comparison.

Reads the CSVs produced by extract_v95_v96_evidence_chain.py plus the held-out
results.json files, and writes five figures + a tables.md into
docs/analysis/traceaad_v95_v96_evidence_chain/.

Usage: uv run python -m experiments.analysis.plot_v95_v96_evidence_chain
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
IN_DIR = REPO / "experiments" / "analysis" / "logs_v95_v96_evidence_chain"
OUT_DIR = REPO / "docs" / "analysis" / "traceaad_v95_v96_evidence_chain"

TASK_ORDER = ["op_aco", "online_bin_packing", "tsp_construct", "cvrp_aco"]
TASK_LABEL = {
    "op_aco": "OP (maximize)",
    "online_bin_packing": "OBP (maximize -bins)",
    "tsp_construct": "TSP (maximize -len)",
    "cvrp_aco": "CVRP (maximize -len)",
}
COLORS = {"v9.5": "#1f77b4", "v9.6": "#d62728"}
E_ALIGN = 900  # all complete runs reach >= 917 real evaluator calls


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    cand = pd.read_csv(IN_DIR / "candidates.csv")
    hist = pd.read_csv(IN_DIR / "history_events.csv")
    return cand, hist


# --------------------------------------------------------------------------- #
# Figure 1: best-so-far vs real evaluator calls
# --------------------------------------------------------------------------- #
def fig_best_so_far(cand: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, task in zip(axes.flat, TASK_ORDER):
        for (ver, rep), d in cand[cand.task == task].groupby(["version", "rep"]):
            d = d.sort_values("sample_order")
            d = d[d.best_so_far.notna()]
            incomplete = d.run_status.iloc[0] != "finished"
            ax.plot(
                d.cum_eval_calls,
                d.best_so_far,
                color=COLORS[ver],
                lw=1.2,
                alpha=0.8,
                ls=":" if incomplete else "-",
                label=f"{ver}" if rep == 1 or (task, ver, rep) in {} else None,
            )
        # y-limits: zoom past the initial ramp
        dd = cand[(cand.task == task) & (cand.cum_eval_calls >= 50)]
        lo, hi = dd.best_so_far.quantile(0.02), dd.best_so_far.max()
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo))
        ax.set_title(TASK_LABEL[task], fontsize=11)
        ax.set_xlabel("real evaluator calls")
        ax.set_ylabel("best-so-far fitness")
        ax.axvline(E_ALIGN, color="gray", lw=0.7, ls="--")
        ax.grid(alpha=0.3)
    handles = [
        plt.Line2D([0], [0], color=COLORS["v9.5"], lw=2, label="V9.5 (3 reps)"),
        plt.Line2D([0], [0], color=COLORS["v9.6"], lw=2, label="V9.6 (3 reps)"),
        plt.Line2D([0], [0], color="gray", lw=1.2, ls=":", label="incomplete run"),
        plt.Line2D([0], [0], color="gray", lw=0.7, ls="--", label=f"E={E_ALIGN} align point"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Best-so-far vs real evaluator calls (per repeat)", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT_DIR / "fig1_best_so_far.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2: history composition (shown counts + pool-vs-shown improve share)
# --------------------------------------------------------------------------- #
def fig_history_composition(hist: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    g = hist.groupby(["task", "version"])
    comp = g.agg(
        formation=("n_formation_shown", "mean"), direct=("n_direct_shown", "mean")
    )
    absent = g.apply(
        lambda d: ((d.n_formation_pool > 0) & (d.n_formation_shown == 0)).sum()
        / max((d.n_formation_pool > 0).sum(), 1),
        include_groups=False,
    )

    ax = axes[0]
    x = np.arange(len(TASK_ORDER))
    width = 0.35
    for off, ver in ((-width / 2, "v9.5"), (width / 2, "v9.6")):
        f = [comp.loc[(t, ver), "formation"] for t in TASK_ORDER]
        d = [comp.loc[(t, ver), "direct"] for t in TASK_ORDER]
        ax.bar(x + off, f, width, color=COLORS[ver], alpha=0.85,
               label=f"{ver} formation")
        ax.bar(x + off, d, width, bottom=f, color=COLORS[ver], alpha=0.35,
               label=f"{ver} direct")
        for i, t in enumerate(TASK_ORDER):
            ax.text(x[i] + off, f[i] + d[i] + 0.1,
                    f"miss\n{absent.loc[(t, ver)]:.0%}", ha="center", fontsize=8)
    ax.set_xticks(x, [TASK_LABEL[t].split(" ")[0] for t in TASK_ORDER])
    ax.set_ylabel("avg events shown (of 8)")
    ax.set_ylim(0, 9.6)
    ax.set_title("Shown history composition (dark=formation, light=direct); "
                 "'miss' = formation absent while formation pool non-empty")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    shares = g.apply(
        lambda d: pd.Series(
            {
                "pool": (d.n_form_pool_improve.sum() + d.n_dir_pool_improve.sum())
                / max(d.n_formation_pool.sum() + d.n_direct_pool.sum(), 1),
                "shown": (d.n_form_shown_improve.sum() + d.n_dir_shown_improve.sum())
                / max(d.n_formation_shown.sum() + d.n_direct_shown.sum(), 1),
            }
        ),
        include_groups=False,
    )
    for off, ver in ((-width / 2, "v9.5"), (width / 2, "v9.6")):
        pool = [shares.loc[(t, ver), "pool"] for t in TASK_ORDER]
        shown = [shares.loc[(t, ver), "shown"] for t in TASK_ORDER]
        ax.bar(x + off, pool, width, color=COLORS[ver], alpha=0.3,
               label=f"{ver} pool (true history)")
        ax.bar(x + off, shown, width * 0.5, color=COLORS[ver], alpha=1.0,
               label=f"{ver} shown to LLM")
    ax.set_xticks(x, [TASK_LABEL[t].split(" ")[0] for t in TASK_ORDER])
    ax.set_ylabel("improve share")
    ax.set_title("Improve share: full local pool (wide, light) vs events shown in "
                 "prompt (narrow, solid)")
    ax.legend(ncol=4, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_history_composition.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3: generation outcome distribution
# --------------------------------------------------------------------------- #
def fig_outcomes(cand: pd.DataFrame) -> None:
    A = cand[cand.iteration.notna()]
    cats = ["improve", "plateau", "regress", "no_op", "invalid", "duplicate"]
    cat_colors = ["#2ca02c", "#98df8a", "#d62728", "#c7c7c7", "#7f7f7f", "#17becf"]
    rates = (
        A.groupby(["task", "version"]).outcome.value_counts(normalize=True)
        .unstack().reindex(columns=cats).fillna(0)
    )
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(TASK_ORDER))
    width = 0.35
    for off, ver in ((-width / 2, "v9.5"), (width / 2, "v9.6")):
        bottom = np.zeros(len(TASK_ORDER))
        for cat, col in zip(cats, cat_colors):
            vals = np.array([rates.loc[(t, ver), cat] for t in TASK_ORDER])
            ax.bar(x + off, vals, width, bottom=bottom, color=col,
                   label=cat if ver == "v9.5" else None)
            bottom += vals
        for i in range(len(TASK_ORDER)):
            ax.text(x[i] + off, 1.015, ver, ha="center", fontsize=7.5, rotation=0)
    ax.set_xticks(x, [TASK_LABEL[t].split(" ")[0] for t in TASK_ORDER])
    ax.set_ylabel("share of anchor-step generations")
    ax.set_title("Generation outcome distribution (formal anchor steps; left bar = V9.5, right = V9.6)")
    ax.legend(ncol=6, fontsize=9, frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, -0.32))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_outcomes.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4: delta-fitness (child - anchor) empirical CDF
# --------------------------------------------------------------------------- #
def fig_delta(cand: pd.DataFrame) -> None:
    A = cand[(cand.iteration.notna()) & cand.delta.notna()]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, task in zip(axes.flat, TASK_ORDER):
        d_task = A[A.task == task]
        lo, hi = d_task.delta.quantile(0.02), d_task.delta.quantile(0.995)
        for ver in ("v9.5", "v9.6"):
            dd = np.sort(d_task[d_task.version == ver].delta.values)
            ax.plot(np.clip(dd, lo, hi), np.linspace(0, 1, len(dd)),
                    color=COLORS[ver], lw=1.8, label=ver)
            med = np.median(dd)
            ax.axvline(med, color=COLORS[ver], lw=0.8, ls=":")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(TASK_LABEL[task], fontsize=11)
        ax.set_xlabel("delta fitness (child - anchor)")
        ax.set_ylabel("CDF")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("Delta-fitness distribution of anchor-step generations "
                 "(dotted line = median; right of 0 = improvement)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_delta_cdf.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 5: conditional on direct-history richness of the anchor
# --------------------------------------------------------------------------- #
def fig_conditional(cand: pd.DataFrame) -> None:
    A = cand[cand.iteration.notna()].copy()
    A["dir_bin"] = pd.cut(
        A.n_direct_pool, [-0.5, 0.5, 2.5, 7.5, np.inf], labels=["0", "1-2", "3-7", ">=8"]
    )
    bins = ["0", "1-2", "3-7", ">=8"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), sharex="col")
    for col, task in enumerate(TASK_ORDER):
        d_task = A[A.task == task]
        ax = axes[0, col]
        ax2 = axes[1, col]
        x = np.arange(len(bins))
        width = 0.38
        for off, ver in ((-width / 2, "v9.5"), (width / 2, "v9.6")):
            dv = d_task[d_task.version == ver]
            imp, med, ns = [], [], []
            for b in bins:
                db = dv[dv.dir_bin == b]
                ns.append(len(db))
                imp.append((db.outcome == "improve").mean() if len(db) else np.nan)
                sc = db[db.delta.notna()]
                med.append(sc.delta.median() if len(sc) else np.nan)
            ax.bar(x + off, imp, width, color=COLORS[ver], alpha=0.85, label=ver)
            for xi, (n_, v_) in enumerate(zip(ns, imp)):
                if n_ > 0 and not np.isnan(v_):
                    ax.text(x[xi] + off, v_ + 0.005, f"{n_}", ha="center",
                            fontsize=6.5, rotation=90)
            ax2.bar(x + off, med, width, color=COLORS[ver], alpha=0.85)
        ax.set_title(TASK_LABEL[task].split(" ")[0], fontsize=11)
        ax.set_ylabel("improve rate" if col == 0 else "")
        ax2.set_ylabel("median delta" if col == 0 else "")
        ax2.set_xticks(x, bins)
        ax2.set_xlabel("# prior direct attempts at anchor")
        if task == "online_bin_packing":
            ax2.set_yscale("symlog", linthresh=1)
        for a in (ax, ax2):
            a.grid(axis="y", alpha=0.3)
        if col == 0:
            ax.legend(fontsize=9)
    fig.suptitle("Conditional on anchor direct-history richness "
                 "(numbers above bars = sample count)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_conditional.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Held-out aggregation
# --------------------------------------------------------------------------- #
def load_heldout() -> pd.DataFrame:
    specs = [
        ("op_aco", "traceaad_v9_5", "eval_best_20260812_v95", "v9.5"),
        ("op_aco", "traceaad_v9_6", "eval_best_20260812_191011", "v9.6"),
        ("online_bin_packing", "traceaad_v9_5", "eval_best_20260812_v95", "v9.5"),
        ("online_bin_packing", "traceaad_v9_6", "eval_best_20260812_191011", "v9.6"),
        ("tsp_construct", "traceaad_v9_5", "eval_best_20260812_v95", "v9.5"),
        ("tsp_construct", "traceaad_v9_6", "eval_best_20260812_191011", "v9.6"),
        ("cvrp_aco", "traceaad_v9_5", "eval_best_20260812_v95", "v9.5"),
        ("cvrp_aco", "traceaad_v9_6", "eval_best_20260812_191011", "v9.6"),
    ]
    rows = []
    for task, mdir, edir, ver in specs:
        path = REPO / "experiments" / task / mdir / edir / "results.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        split_map = (
            d.get("results_by_split")
            or d.get("eval_results_by_scale")
            or d.get("eval_results_by_size")
        )
        for split, sub in split_map.items():
            for r in sub["results"]:
                rows.append(
                    {
                        "task": task,
                        "version": ver,
                        "split": str(split),
                        "run_name": r["run_name"],
                        "eval_score": r["eval_score"],
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def write_tables(cand: pd.DataFrame, hist: pd.DataFrame) -> None:
    lines: list[str] = ["# V9.5 vs V9.6 aggregate tables (auto-generated)\n"]

    def md(df: pd.DataFrame, title: str) -> None:
        lines.append(f"## {title}\n")
        lines.append("```text")
        lines.append(df.round(4).to_string())
        lines.append("```")
        lines.append("")

    # ---- layer 1
    g = hist.groupby(["task", "version"])
    layer1 = g.agg(
        events=("n_shown", "size"),
        formation_shown=("n_formation_shown", "mean"),
        direct_shown=("n_direct_shown", "mean"),
        direct_pool=("n_direct_pool", "mean"),
        prompt_tokens=("hist_prompt_tokens", "mean"),
    )
    layer1["formation_absent"] = g.apply(
        lambda d: ((d.n_formation_pool > 0) & (d.n_formation_shown == 0)).sum()
        / max((d.n_formation_pool > 0).sum(), 1),
        include_groups=False,
    )
    layer1["pool_improve_share"] = g.apply(
        lambda d: (d.n_form_pool_improve.sum() + d.n_dir_pool_improve.sum())
        / max(d.n_formation_pool.sum() + d.n_direct_pool.sum(), 1),
        include_groups=False,
    )
    layer1["shown_improve_share"] = g.apply(
        lambda d: (d.n_form_shown_improve.sum() + d.n_dir_shown_improve.sum())
        / max(d.n_formation_shown.sum() + d.n_direct_shown.sum(), 1),
        include_groups=False,
    )
    md(layer1, "Layer 1: history composition per generation")

    # ---- layer 2
    A = cand[cand.iteration.notna()]
    rates = (
        A.groupby(["task", "version"]).outcome.value_counts(normalize=True)
        .unstack().fillna(0)
    )
    md(rates, "Layer 2: outcome rates (anchor steps)")
    S = A[A.delta.notna()]

    def dstats(d: pd.DataFrame) -> pd.Series:
        pos = d.delta[d.delta > 0]
        neg = d.delta[d.delta < 0]
        return pd.Series(
            {
                "n": len(d),
                "mean_delta": d.delta.mean(),
                "median_delta": d.delta.median(),
                "P(delta>0)": (d.delta > 0).mean(),
                "mean_pos": pos.mean() if len(pos) else np.nan,
                "mean_neg": neg.mean() if len(neg) else np.nan,
                "P(new_best)": d.is_new_best.mean(),
                "median_changed_lines": d.changed_lines.median(),
            }
        )

    md(
        S.groupby(["task", "version"]).apply(dstats, include_groups=False),
        "Layer 2: delta-fitness statistics (scored anchor steps)",
    )

    # ---- conditional
    Ac = A.copy()
    Ac["dir_bin"] = pd.cut(
        Ac.n_direct_pool, [-0.5, 0.5, 2.5, 7.5, np.inf],
        labels=["0", "1-2", "3-7", ">=8"],
    )
    cond = Ac.groupby(["task", "dir_bin", "version"], observed=True).apply(
        lambda d: pd.Series(
            {
                "n": len(d),
                "improve_rate": (d.outcome == "improve").mean(),
                "median_delta": d[d.delta.notna()].delta.median(),
            }
        ),
        include_groups=False,
    )
    md(cond, "Conditional: by # prior direct attempts at anchor")

    # ---- layer 3
    rows = []
    for (task, ver, rep), d in cand.groupby(["task", "version", "rep"]):
        d = d.sort_values("sample_order")
        calls = d.cum_eval_calls.values
        best = d.best_so_far.values
        maxc = calls.max()
        ck = {}
        for E in (100, 200, 300, 500, 750, 900):
            mask = calls <= E
            ck[f"E{E}"] = best[mask][-1] if mask.any() else np.nan
        auc = np.nan
        if maxc >= E_ALIGN:
            b0 = None
            prev_c = 0
            area = 0.0
            for c, b in zip(calls, best):
                if np.isnan(b):
                    continue
                if c > E_ALIGN:
                    break
                if b0 is not None and c > prev_c:
                    area += b0 * (c - prev_c)
                b0 = b
                prev_c = max(prev_c, c)
            area += b0 * (E_ALIGN - prev_c)
            auc = area / E_ALIGN
        nb = d[d.is_new_best == 1]
        rows.append(
            dict(
                task=task,
                version=ver,
                rep=rep,
                max_calls=maxc,
                status=d.run_status.iloc[0],
                **ck,
                auc900=auc,
                final_best=best[~np.isnan(best)][-1],
                last_best_call=nb.cum_eval_calls.max(),
                n_best_updates=len(nb),
            )
        )
    layer3 = pd.DataFrame(rows).set_index(["task", "version", "rep"]).sort_index()
    md(layer3, "Layer 3: per-run best-so-far checkpoints / AUC / stagnation")

    # ---- layer 4 held-out
    H = load_heldout()
    if len(H):
        piv = H.groupby(["task", "split", "version"]).eval_score.agg(
            ["count", "mean", "std", "min", "max"]
        )
        md(piv, "Layer 4: held-out eval_score by split (higher is better)")
        detail = H.pivot_table(
            index=["task", "split", "version"], columns="run_name",
            values="eval_score", aggfunc="first",
        )
        lines.append("## Layer 4 detail: per-run held-out scores\n")
        for (task, split), d in H.groupby(["task", "split"]):
            lines.append(f"- **{task} / {split}**")
            for ver, dv in d.groupby("version"):
                vals = ", ".join(
                    f"{r.run_name.split('_')[-1]}={r.eval_score:.4f}"
                    for r in dv.itertuples()
                )
                lines.append(f"  - {ver}: {vals}")
        lines.append("")

    (OUT_DIR / "tables.md").write_text("\n".join(lines))
    print(f"wrote {OUT_DIR / 'tables.md'}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cand, hist = load()
    fig_best_so_far(cand)
    fig_history_composition(hist)
    fig_outcomes(cand)
    fig_delta(cand)
    fig_conditional(cand)
    write_tables(cand, hist)
    for f in sorted(OUT_DIR.glob("*.png")):
        print("wrote", f)


if __name__ == "__main__":
    main()

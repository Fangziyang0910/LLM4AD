"""Frozen task-internal proxy mechanism tags and proxy-region frontier view.

``mechanism_tags`` / ``macro_family`` are the static lexical rules frozen from
the V9.7 batch ``20260814_150927`` post-hoc analyses (search geometry and
region revisit).  They read the code after dropping comments and string
literals, never read the declared Idea, and are not modified during runs.
They are an auditable proxy, not semantic ground truth.

``RegionView`` compresses the programs whose real evaluation has completed
into per-proxy-region frontier rows.  Callers must feed programs in real
evaluation order; the online search does so because responses are processed
strictly sequentially before the next decision.
"""

from __future__ import annotations

import io
import re
import tokenize
from collections import Counter
from dataclasses import dataclass
from typing import Any, Final, Iterable

from .history import format_fitness
from .schema import Program

PROXY_RULES_VERSION: Final[str] = "v97-search-geometry-20260814-frozen"
PROXY_TASKS: Final[frozenset[str]] = frozenset(
    {"tsp_construct", "cvrp_aco", "op_aco", "online_bin_packing"}
)


def code_tokens(code: str) -> tuple[str, Counter[str]]:
    """Return code without comments/strings plus identifier counts."""
    kept: list[str] = []
    names: Counter[str] = Counter()
    try:
        stream = tokenize.generate_tokens(io.StringIO(code).readline)
        for token in stream:
            if token.type in {
                tokenize.COMMENT,
                tokenize.STRING,
                tokenize.ENCODING,
                tokenize.ENDMARKER,
            }:
                continue
            value = token.string.lower()
            kept.append(value)
            if token.type == tokenize.NAME:
                names[value] += 1
    except (IndentationError, tokenize.TokenError):
        lowered = code.lower()
        kept = re.findall(r"[a-z_][a-z0-9_]*|\S", lowered)
        names.update(re.findall(r"[a-z_][a-z0-9_]*", lowered))
    return " ".join(kept), names


def has_any(text: str, *patterns: str) -> bool:
    return any(re.search(pattern, text) is not None for pattern in patterns)


def mechanism_tags(task: str, code: str) -> frozenset[str]:
    text, names = code_tokens(code)
    tags: set[str] = set()

    if task == "tsp_construct":
        if has_any(text, r"lru_cache", r"remaining_mask", r"bitmask"):
            tags.add("exact_dp")
        if has_any(text, r"beam_search", r"beam_width", r"\bbeam\b"):
            tags.add("beam_search")
        if has_any(
            text,
            r"completion_cost",
            r"temp_remaining",
            r"sampled_nn",
            r"nearest_neighbor.*completion",
            r"simulate.*tour",
        ):
            tags.add("tour_rollout")
        if has_any(text, r"2_opt", r"2opt", r"cost_original"):
            tags.add("two_opt")
        if has_any(text, r"argpartition", r"top_k", r"candidates_to_check"):
            tags.add("candidate_screening")
        if has_any(
            text,
            r"dist_unvisited",
            r"forward_cost",
            r"connectivity",
            r"isolation",
            r"future_cost",
        ):
            tags.add("one_step_lookahead")
        if has_any(text, r"remaining_count", r"progress", r"stage_factor"):
            tags.add("stage_adaptive")
        if names["destination_node"] > 1:
            tags.add("destination_aware")

    elif task == "cvrp_aco":
        if has_any(text, r"cluster_ids", r"sorted_customer_indices"):
            tags.add("sweep_partition")
        if has_any(text, r"clarke", r"saving", r"savings"):
            tags.add("clarke_wright_savings")
        if has_any(text, r"arctan2", r"angle_diff", r"angular"):
            tags.add("angular_geometry")
        if has_any(text, r"radial", r"dist_to_depot", r"depot_distance"):
            tags.add("depot_radial")
        if names["demands"] > 1 or names["capacity"] > 1:
            tags.add("demand_capacity")
        if has_any(text, r"feasible_mask", r"infeasible_mask", r"cap_factor"):
            tags.add("pair_feasibility")
        if has_any(text, r"neighbor", r"cluster_bonus", r"density"):
            tags.add("neighborhood_density")

    elif task == "op_aco":
        if names["prize"] > 1 and names["distance"] > 1:
            tags.add("prize_distance")
        if has_any(
            text,
            r"round_trip",
            r"return.*depot",
            r"feasible_mask",
            r"dist_j_to_depot",
        ):
            tags.add("return_feasibility")
        if has_any(
            text,
            r"remaining_budget",
            r"budget_factor",
            r"cost_pressure",
            r"budget_tightness",
        ):
            tags.add("budget_pressure")
        if has_any(text, r"top_k_indices", r"target_scores", r"directional_factor"):
            tags.add("directional_targets")
        if has_any(
            text,
            r"cluster_prize",
            r"near_prizes",
            r"forward_factor",
            r"normalized_potential",
        ):
            tags.add("neighborhood_potential")
        if has_any(text, r"softmax", r"exp_scores", r"np \. exp"):
            tags.add("nonlinear_emphasis")

    elif task == "online_bin_packing":
        if has_any(text, r"global\s+_", r"item_history", r"history"):
            tags.add("stateful_history")
        if has_any(text, r"np \. mean", r"np \. std", r"percentile", r"pdf_score"):
            tags.add("distribution_model")
        if has_any(text, r"probab", r"likelihood", r"future", r"fit_score"):
            tags.add("future_fit")
        if has_any(
            text,
            r"fragment",
            r"awkward",
            r"closure_bonus",
            r"gap_penalty",
            r"tiny_threshold",
        ):
            tags.add("fragmentation_control")
        if has_any(text, r"diversity", r"spread", r"balance"):
            tags.add("balance_spread")
        if has_any(text, r"remaining_after", r"leftover", r"remainder"):
            tags.add("best_fit")
        if has_any(text, r"threshold", r"np \. where"):
            tags.add("piecewise_bands")
    else:
        raise ValueError(f"unknown task: {task}")
    return frozenset(tags)


def macro_family(task: str, tags: frozenset[str]) -> str:
    if task == "tsp_construct":
        if tags & {"exact_dp", "beam_search"}:
            return "explicit_search"
        if tags & {"tour_rollout", "two_opt"}:
            return "completion_rollout"
        if "one_step_lookahead" in tags:
            return "lookahead_score"
        return "local_score"
    if task == "cvrp_aco":
        if "sweep_partition" in tags:
            return "sweep_partition"
        if "clarke_wright_savings" in tags:
            return "savings_geometry"
        if "angular_geometry" in tags and "neighborhood_density" in tags:
            return "spatial_cluster"
        if "angular_geometry" in tags:
            return "angular_radial"
        if "demand_capacity" in tags:
            return "capacity_distance"
        return "distance_only"
    if task == "op_aco":
        if "directional_targets" in tags:
            return "target_direction"
        if "neighborhood_potential" in tags:
            return "neighborhood_potential"
        if {"return_feasibility", "budget_pressure"} <= tags:
            return "budget_feasible_density"
        if "return_feasibility" in tags:
            return "feasibility_density"
        return "prize_distance"
    if task == "online_bin_packing":
        if {"stateful_history", "distribution_model"} <= tags:
            return "online_distribution"
        if "distribution_model" in tags:
            return "distribution_fit"
        if "future_fit" in tags:
            return "future_gap_model"
        if "fragmentation_control" in tags:
            return "fragmentation_best_fit"
        if "balance_spread" in tags:
            return "balance_fit"
        return "best_fit"
    raise ValueError(f"unknown task: {task}")


@dataclass(frozen=True, slots=True)
class FrontierProgram:
    program_id: int
    family: str
    tags: frozenset[str]
    q: float
    fitness: float
    code: str
    length: int
    eval_index: int


class RegionView:
    """Per-proxy-region frontier over programs whose evaluation completed.

    Recording is idempotent by program id.  ``eval_index`` is the arrival
    order of ``record`` calls; callers feed programs in real evaluation
    order so exact ties prefer the earlier-evaluated program.
    """

    def __init__(self, task: str) -> None:
        if task not in PROXY_TASKS:
            raise ValueError(f"unknown proxy task: {task}")
        self.task = task
        self._recorded: set[int] = set()
        self._best: dict[str, FrontierProgram] = {}
        self._count = 0

    def record(self, program: Program) -> bool:
        if program.id in self._recorded:
            return False
        self._recorded.add(program.id)
        self._count += 1
        tags = mechanism_tags(self.task, program.code)
        family = macro_family(self.task, tags)
        current = self._best.get(family)
        candidate = FrontierProgram(
            program_id=program.id,
            family=family,
            tags=tags,
            q=program.q,
            fitness=program.fitness,
            code=program.code,
            length=program.length,
            eval_index=self._count,
        )
        if current is None or _frontier_key(candidate) > _frontier_key(current):
            self._best[family] = candidate
        return True

    def recorded_count(self) -> int:
        return len(self._recorded)

    def visited_families(self) -> tuple[str, ...]:
        return tuple(self._best)

    def frontier_program(self, family: str) -> FrontierProgram:
        return self._best[family]

    def frontier_rows(self) -> tuple[FrontierProgram, ...]:
        return tuple(
            sorted(self._best.values(), key=lambda row: (-row.q, row.family))
        )

    def frontier_of_code(self, code: str) -> tuple[frozenset[str], str]:
        tags = mechanism_tags(self.task, code)
        return tags, macro_family(self.task, tags)

    def global_best_q(self) -> float | None:
        rows = self.frontier_rows()
        return rows[0].q if rows else None

    def frontier_text(self, anchor_family: str) -> str:
        rows = [
            {"family": row.family, "tags": row.tags, "q": row.q}
            for row in self.frontier_rows()
        ]
        return render_frontier_table(rows, anchor_family, self.global_best_q())


def render_frontier_table(
    rows: Iterable[dict[str, Any]], anchor_family: str, global_best_q: float | None
) -> str:
    """Render the r3 floor table: the anchor's own region first, every
    searched region's frontier shown with its mechanism tags and quality
    under floor semantics (recorded levels, no code or counts)."""

    lines = [
        "[Searched Proxy Regions]",
        (
            "Earlier in this search the following proxy mechanism regions "
            "were already implemented and evaluated. A candidate that merely "
            "rebuilds a region below its recorded level wastes budget."
        ),
    ]
    ordered = list(rows)
    own = [row for row in ordered if row["family"] == anchor_family]
    others = sorted(
        (row for row in ordered if row["family"] != anchor_family),
        key=lambda row: (-float(row["q"]), str(row["family"])),
    )
    if own:
        row = own[0]
        lines.append("")
        lines.append("[Current Algorithm's Region]")
        lines.append(
            f"Observed tags of frontier program: {_format_tags(row['tags'])}"
        )
        lines.append(f"Directed quality: {format_fitness(float(row['q']))}")
    lines.append("")
    lines.append("[Other Searched Regions]")
    if others:
        for index, row in enumerate(others, start=2):
            lines.append(
                f"Region {index} observed tags of frontier program: "
                f"{_format_tags(row['tags'])}"
            )
            lines.append(
                f"Region {index} directed quality: {format_fitness(float(row['q']))}"
            )
    else:
        lines.append("No other regions have been searched yet.")
    if global_best_q is not None:
        lines.append("")
        lines.append(
            "Global best directed quality across all regions: "
            f"{format_fitness(float(global_best_q))}"
        )
    return "\n".join(lines)


def _frontier_key(row: FrontierProgram) -> tuple[float, int, int]:
    # q desc, then shorter code, then earlier evaluation.
    return (row.q, -row.length, -row.eval_index)


def _format_tags(tags: frozenset[str]) -> str:
    return ", ".join(sorted(tags)) if tags else "none"


def build_region_view(task: str, programs: Iterable[Program]) -> RegionView:
    view = RegionView(task)
    for program in programs:
        view.record(program)
    return view


__all__ = [
    "PROXY_RULES_VERSION",
    "PROXY_TASKS",
    "FrontierProgram",
    "RegionView",
    "build_region_view",
    "code_tokens",
    "has_any",
    "macro_family",
    "mechanism_tags",
    "render_frontier_table",
]

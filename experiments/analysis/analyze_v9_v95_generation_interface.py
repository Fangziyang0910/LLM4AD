#!/usr/bin/env python3
"""Audit the generation interfaces of TraceAAD V9 and V9.5.

The script uses every completed formal run for aggregate request/proposal metrics and
draws a deterministic 20-prompt sample per task and version for prompt decomposition.
V9.5 prompts are read verbatim. V9 prompts are reconstructed from the recorded parent,
operator, reference and exact edge-id snapshot in ``decisions.jsonl``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.runners._common import (  # noqa: E402
    BACKENDS,
    build_task,
    resolve_llm_api_key,
)
from llm4ad.base import TextFunctionProgramConverter  # noqa: E402
from llm4ad.method.traceaad_v9.complexity import code_change_ratio  # noqa: E402
from llm4ad.method.traceaad_v9.context import build_code_prompt  # noqa: E402
from llm4ad.method.traceaad_v9.operators import DEFAULT_OPERATORS  # noqa: E402
from llm4ad.method.traceaad_v9.prompt import format_fitness  # noqa: E402
from llm4ad.method.traceaad_v9.schema import (  # noqa: E402
    ImprovementEdge,
    OperatorName,
    ProgramNode,
    VirtualRoot,
)
from llm4ad.method.traceaad_v9.tree import SearchTree, is_node_better  # noqa: E402


TASKS = {
    "tsp": "tsp_construct",
    "cvrp": "cvrp_aco",
    "op": "op_aco",
}
V9_BATCH = "20260807_123753"
V95_BATCH = "20260811_171029"
SAMPLE_SIZE = 20
SAMPLE_SEED = 9509


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def mean(values: list[float]) -> float | None:
    return None if not values else statistics.fmean(values)


def median(values: list[float]) -> float | None:
    return None if not values else statistics.median(values)


def run_dirs(task: str, version: str) -> list[Path]:
    task_dir = TASKS[task]
    if version == "v9":
        root = PROJECT_ROOT / "experiments" / task_dir / "traceaad_v9" / "version9"
        pattern = f"v9_{V9_BATCH}_v9_{task}_rep*"
    else:
        root = PROJECT_ROOT / "experiments" / task_dir / "traceaad_v9_5"
        pattern = f"v9_5_{V95_BATCH}_{task}_rep*"
    runs = sorted(path for path in root.glob(pattern) if path.is_dir())
    return [
        path
        for path in runs
        if load_json(path / "logs" / "summary.json").get("status") == "finished"
    ]


def search_llm_calls(run_dir: Path, version: str) -> list[dict[str, Any]]:
    stage = "direct_code" if version == "v9" else "search"
    rows = load_jsonl(run_dir / "artifacts" / "llm_calls.jsonl")
    return [
        row
        for row in rows
        if row.get("stage") == stage and row.get("status") != "transport"
    ]


def snapshot_tree(checkpoint: dict[str, Any], cutoff: int) -> SearchTree:
    payload = checkpoint["tree"]
    eligible = {
        int(item["id"]): item
        for item in payload["nodes"]
        if int(item["creation_order"]) < cutoff
    }
    eligible_edges = {
        int(item["id"]): item
        for item in payload["edges"]
        if int(item["sample_order"]) < cutoff
    }
    tree = SearchTree()
    tree.root = VirtualRoot(
        id=-1,
        child_ids=[
            int(item) for item in payload["root"]["child_ids"] if int(item) in eligible
        ],
        visit_count=0,
    )
    for node_id, item in eligible.items():
        tree._nodes[node_id] = ProgramNode(
            id=node_id,
            code=str(item["code"]),
            idea=str(item["idea"]),
            fitness=float(item["fitness"]),
            directed_fitness=float(item["directed_fitness"]),
            program_loc=int(item["program_loc"]),
            code_hash=str(item["code_hash"]),
            parent_id=int(item["parent_id"]),
            incoming_edge_id=(
                None
                if item["incoming_edge_id"] is None
                else int(item["incoming_edge_id"])
            ),
            child_ids=[int(child) for child in item["child_ids"] if int(child) in eligible],
            depth=int(item["depth"]),
            visit_count=int(item["visit_count"]),
            expansion_count=int(item["expansion_count"]),
            subtree_value=float(item["directed_fitness"]),
            subtree_best_node_id=node_id,
            creation_order=int(item["creation_order"]),
            batch_id=None if item["batch_id"] is None else int(item["batch_id"]),
            operator=str(item["operator"]),
        )
    for edge_id, item in eligible_edges.items():
        tree._edges[edge_id] = ImprovementEdge(
            id=edge_id,
            parent_id=int(item["parent_id"]),
            child_id=int(item["child_id"]),
            operator=OperatorName(item["operator"]),
            implemented_idea=str(item["implemented_idea"]),
            reference_node_id=(
                None
                if item["reference_node_id"] is None
                else int(item["reference_node_id"])
            ),
            reference_root_branch_id=(
                None
                if item["reference_root_branch_id"] is None
                else int(item["reference_root_branch_id"])
            ),
            delta_parent=float(item["delta_parent"]),
            delta_global_best=(
                None
                if item["delta_global_best"] is None
                else float(item["delta_global_best"])
            ),
            outcome=str(item["outcome"]),
            delta_loc=int(item["delta_loc"]),
            code_change_ratio=float(item["code_change_ratio"]),
            new_global_best=bool(item["new_global_best"]),
            global_best_update_reason=item["global_best_update_reason"],
            iteration=int(item["iteration"]),
            batch_id=int(item["batch_id"]),
            sibling_seq=int(item["sibling_seq"]),
            sample_order=int(item["sample_order"]),
        )

    def update(node_id: int) -> ProgramNode:
        node = tree.get_node(node_id)
        best = node
        for child_id in node.child_ids:
            candidate = update(child_id)
            if is_node_better(candidate, best):
                best = candidate
        node.subtree_value = best.directed_fitness
        node.subtree_best_node_id = best.id
        return best

    root_best: ProgramNode | None = None
    for root_id in tree.root.child_ids:
        candidate = update(root_id)
        if is_node_better(candidate, root_best):
            root_best = candidate
    tree.root.subtree_best_node_id = None if root_best is None else root_best.id
    tree.root.subtree_value = None if root_best is None else root_best.directed_fitness
    return tree


def one_line(text: str, limit: int) -> str:
    compact = " ".join(str(text).split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


def formation_text(tree: SearchTree, edge_ids: list[int]) -> str:
    lines = ["[How This Program Was Reached]"]
    if not edge_ids:
        lines.append("This is an initial program with no previous changes.")
    for position, edge_id in enumerate(edge_ids, start=1):
        edge = tree.get_edge(edge_id)
        parent = tree.get_node(edge.parent_id)
        child = tree.get_node(edge.child_id)
        lines.extend(
            [
                (
                    f"Step {position}: {edge.outcome}; fitness "
                    f"{format_fitness(parent.fitness)} -> {format_fitness(child.fitness)}; "
                    f"global breakthrough={'yes' if edge.new_global_best else 'no'}"
                ),
                f"  Implemented change: {one_line(edge.implemented_idea, 360)}",
                (
                    f"  Code change: {edge.code_change_ratio:.0%}; "
                    f"LOC {parent.program_loc} -> {child.program_loc}"
                ),
            ]
        )
    return "\n".join(lines)


def current_history_text(
    tree: SearchTree, formation_ids: list[int], direct_ids: list[int]
) -> str:
    lines = [formation_text(tree, formation_ids), "[Previously Tested From This Program]"]
    if not direct_ids:
        lines.append("No direct modifications have been evaluated from this program yet.")
    for position, edge_id in enumerate(direct_ids, start=1):
        edge = tree.get_edge(edge_id)
        child = tree.get_node(edge.child_id)
        best = tree.subtree_best(child.id)
        lines.extend(
            [
                (
                    f"Branch {position}: {edge.outcome}; immediate fitness "
                    f"{format_fitness(child.fitness)}; subtree-best fitness "
                    f"{format_fitness(best.fitness)}; depth to subtree best "
                    f"{best.depth - child.depth}"
                ),
                f"  Implemented change: {one_line(child.idea, 360)}",
            ]
        )
    return "\n".join(lines)


@lru_cache(maxsize=None)
def task_prompt_inputs(task: str):
    evaluation, _ = build_task(task, eval_workers=1)
    template = TextFunctionProgramConverter.text_to_program(evaluation.template_program)
    if template is None:
        raise ValueError(f"cannot parse task template for {task}")
    return evaluation.task_description, template.functions[0]


def reconstruct_v9_prompt(run_dir: Path, call: dict[str, Any]) -> str:
    checkpoint = load_json(run_dir / "checkpoints" / "latest.json")
    decisions = load_jsonl(run_dir / "artifacts" / "decisions.jsonl")
    decision = next(
        item
        for item in decisions
        if item.get("event") == "node_selected"
        and item.get("attempt_id") == call.get("iteration")
    )
    calls = search_llm_calls(run_dir, "v9")
    cutoff = min(
        int(item["sample_order"])
        for item in calls
        if item.get("iteration") == call.get("iteration")
    )
    tree = snapshot_tree(checkpoint, cutoff)
    base = tree.get_node(int(decision["selected_node_id"]))
    reference_id = decision.get("reference_node_id")
    reference = None if reference_id is None else tree.get_node(int(reference_id))
    operators = {str(item.name): item() for item in DEFAULT_OPERATORS}
    operator = operators[str(decision["operator"])]
    task = load_json(run_dir / "run_config.json")["task"]
    task_description, target = task_prompt_inputs(task)
    current_history = current_history_text(
        tree,
        [int(item) for item in decision["current_formation_edge_ids"]],
        [int(item) for item in decision["direct_child_edge_ids"]],
    )
    reference_history = (
        ""
        if reference is None
        else formation_text(
            tree, [int(item) for item in decision["reference_formation_edge_ids"]]
        )
    )
    return build_code_prompt(
        current_node=base,
        current_history=current_history,
        operator_constraint=operator.prompt_constraint,
        task_description=task_description,
        template_function=target,
        maximize=True,
        candidate_index=int(call.get("seq", 0)),
        candidate_count=int(decision["requested_children"]),
        reference_node=reference,
        reference_history=reference_history,
    )


def prompt_blocks(version: str, prompt: str) -> list[tuple[str, str]]:
    if version == "v95":
        markers = [
            ("task", 0),
            ("current_program", prompt.index("[Current Executable Anchor]")),
            ("history", prompt.index("[Recent Formation Corrections]")),
            ("instruction", prompt.index("[Instruction]")),
        ]
    else:
        history_start = prompt.index("[How This Program Was Reached]")
        current_start = prompt.index("[Current Program]")
        direction_start = prompt.index("[Improvement Direction]")
        target_start = prompt.index("[Target Function]")
        instruction_start = prompt.index("[Instruction]")
        markers = [("task", 0), ("history", history_start), ("current_program", current_start)]
        if "[Reference Root Branch History]" in prompt:
            ref_history = prompt.index("[Reference Root Branch History]")
            ref_program = prompt.index("[Reference Program]")
            markers.extend(
                [("reference_history", ref_history), ("reference_program", ref_program)]
            )
        markers.extend(
            [
                ("operator_direction", direction_start),
                ("target_signature", target_start),
                ("instruction", instruction_start),
            ]
        )
        markers.sort(key=lambda item: item[1])
    blocks: list[tuple[str, str]] = []
    for index, (label, start) in enumerate(markers):
        end = len(prompt) if index + 1 == len(markers) else markers[index + 1][1]
        blocks.append((label, prompt[start:end]))
    return blocks


class LiveTokenizer:
    def __init__(self) -> None:
        profile = BACKENDS["server3"]
        self.base_url = str(profile.base_url).rstrip("/")
        self.url = (
            self.base_url[: -len("/v1")] + "/tokenize"
            if self.base_url.endswith("/v1")
            else self.base_url + "/tokenize"
        )
        self.model = str(profile.model)
        key = resolve_llm_api_key(base_url=self.base_url)
        self.headers = {} if key == "EMPTY" else {"Authorization": f"Bearer {key}"}

    def count(self, text: str, chat: bool) -> int:
        payload: dict[str, Any]
        if chat:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": text.strip()}],
                "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        else:
            payload = {"model": self.model, "prompt": text}
        error: Exception | None = None
        for _ in range(3):
            try:
                response = requests.post(
                    self.url,
                    json=payload,
                    headers=self.headers,
                    timeout=30,
                )
                response.raise_for_status()
                body = response.json()
                if isinstance(body.get("count"), int):
                    return int(body["count"])
                for key in ("tokens", "token_ids"):
                    if isinstance(body.get(key), list):
                        return len(body[key])
                raise ValueError("tokenizer response omitted token count")
            except Exception as exc:
                error = exc
        raise RuntimeError("live tokenizer failed") from error


def token_decomposition(
    sampled: list[dict[str, Any]], tokenizer: LiveTokenizer
) -> None:
    requests_to_make: set[tuple[bool, str]] = set()
    for item in sampled:
        prefix = ""
        prefixes: list[tuple[str, str]] = []
        for label, block in item["blocks"]:
            prefix += block
            prefixes.append((label, prefix))
            requests_to_make.add((False, prefix))
        item["prefixes"] = prefixes
        requests_to_make.add((True, item["prompt"]))

    cache: dict[tuple[bool, str], int] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(tokenizer.count, text, chat): (chat, text)
            for chat, text in requests_to_make
        }
        for future in as_completed(futures):
            cache[futures[future]] = future.result()

    for item in sampled:
        previous = 0
        component: dict[str, int] = {}
        for label, prefix in item.pop("prefixes"):
            current = cache[(False, prefix)]
            component[label] = component.get(label, 0) + current - previous
            previous = current
        item["recounted_raw_tokens"] = previous
        item["recounted_chat_tokens"] = cache[(True, item["prompt"])]
        item["component_tokens"] = component


def aggregate_request_metrics(runs: list[Path], version: str) -> dict[str, Any]:
    rows = [row for run in runs for row in search_llm_calls(run, version)]
    return {
        "completed_search_requests": len(rows),
        "mean_logged_prompt_tokens_raw": mean([float(row["prompt_tokens"]) for row in rows]),
        "median_logged_prompt_tokens_raw": median([float(row["prompt_tokens"]) for row in rows]),
        "mean_response_tokens_raw": mean([float(row["response_tokens"]) for row in rows]),
        "parse_failure_rate_per_completed_response": mean(
            [1.0 if row.get("status") == "parse_failed" else 0.0 for row in rows]
        ),
    }


def aggregate_context_metrics(runs: list[Path], version: str) -> dict[str, Any]:
    formation: list[float] = []
    direct: list[float] = []
    reference_formation: list[float] = []
    has_reference: list[float] = []
    for run in runs:
        decisions = load_jsonl(run / "artifacts" / "decisions.jsonl")
        if version == "v9":
            for item in decisions:
                if item.get("event") != "node_selected":
                    continue
                weight = int(item["requested_children"])
                formation.extend(
                    [float(len(item["current_formation_edge_ids"]))] * weight
                )
                direct.extend([float(len(item["direct_child_edge_ids"]))] * weight)
                reference_formation.extend(
                    [float(len(item["reference_formation_edge_ids"]))] * weight
                )
                has_reference.extend(
                    [1.0 if item.get("reference_node_id") is not None else 0.0]
                    * weight
                )
        else:
            initial_root_count = int(
                load_json(run / "run_config.json")["method_params"]["initial_root_count"]
            )
            evidence_events = [
                item for item in decisions if item.get("event") == "evidence_built"
            ][initial_root_count:]
            for item in evidence_events:
                formation.append(float(len(item["selected_formation_ids"])))
                direct.append(float(len(item["selected_direct_ids"])))
                reference_formation.append(0.0)
                has_reference.append(0.0)
    total = [a + b + c for a, b, c in zip(formation, direct, reference_formation)]
    return {
        "prompt_count": len(total),
        "mean_formation_events": mean(formation),
        "mean_direct_events": mean(direct),
        "mean_reference_formation_events": mean(reference_formation),
        "mean_total_history_events": mean(total),
        "reference_program_prompt_rate": mean(has_reference),
    }


def aggregate_generation_metrics(runs: list[Path], version: str) -> dict[str, Any]:
    evaluator_calls: list[int] = []
    budgets: list[int] = []
    valid_hashes: list[str] = []
    valid_count = 0
    improvement = 0
    outcome_count = 0
    change_ratios: list[float] = []
    invalid = 0
    candidate_count = 0

    for run in runs:
        summary = load_json(run / "logs" / "summary.json")
        candidates = load_jsonl(run / "artifacts" / "candidates.jsonl")
        if version == "v9":
            candidates = [item for item in candidates if item.get("operator") != "init"]
        else:
            candidates = [item for item in candidates if item.get("stage") == "search"]
        checkpoint = load_json(run / "checkpoints" / "latest.json")
        evaluator_calls.append(int(summary.get("evaluator_call_count", summary["num_samples"])))
        budgets.append(int(summary.get("candidate_count", summary["num_samples"])))
        candidate_count += len(candidates)
        if version == "v9":
            edges = load_jsonl(run / "artifacts" / "edges.jsonl")
            improvement += sum(item["outcome"] == "improve" for item in edges)
            outcome_count += len(edges)
            change_ratios.extend(float(item["code_change_ratio"]) for item in edges)
            invalid += sum(item.get("status") != "ok" for item in candidates)
            valid = [item for item in candidates if item.get("status") == "ok"]
            valid_count += len(valid)
            valid_hashes.extend(str(item["code_hash"]) for item in valid)
        else:
            forest = checkpoint["forest"]
            states = {int(item["state_id"]): item for item in forest["states"]}
            artifacts = {int(item["artifact_id"]): item for item in forest["artifacts"]}
            invalid += sum(item.get("status") != "ok" for item in candidates)
            valid = [item for item in candidates if item.get("status") == "ok"]
            valid_count += len(valid)
            valid_hashes.extend(
                str(item["evaluator_input_hash"])
                for item in valid
                if item.get("evaluator_input_hash") is not None
            )
            outcomes = [
                item
                for item in candidates
                if item.get("direct_outcome") in {"improve", "plateau", "regress"}
            ]
            improvement += sum(item["direct_outcome"] == "improve" for item in outcomes)
            outcome_count += len(outcomes)
            for item in outcomes:
                state = states[int(item["parent_node_id"])]
                parent = artifacts[int(state["artifact_id"])]["evaluator_input_code"]
                child = str(item["program"])
                change_ratios.append(code_change_ratio(parent, child))

    return {
        "finished_runs": len(runs),
        "budget_unit": "evaluated_candidate" if version == "v9" else "completed_response",
        "mean_nominal_budget": mean([float(item) for item in budgets]),
        "mean_actual_evaluator_calls": mean([float(item) for item in evaluator_calls]),
        "evaluator_calls_by_run": evaluator_calls,
        "invalid_candidate_rate": invalid / candidate_count,
        "parent_improvement_rate_valid_transition": improvement / outcome_count,
        "mean_line_change_ratio": mean(change_ratios),
        "median_line_change_ratio": median(change_ratios),
        "unique_valid_program_rate": len(set(valid_hashes)) / valid_count,
        "valid_program_count": valid_count,
    }


def sample_prompts() -> tuple[list[dict[str, Any]], dict[str, dict[str, list[Path]]]]:
    rng = random.Random(SAMPLE_SEED)
    sampled: list[dict[str, Any]] = []
    inventory: dict[str, dict[str, list[Path]]] = {}
    for task in TASKS:
        inventory[task] = {}
        for version in ("v9", "v95"):
            runs = run_dirs(task, version)
            inventory[task][version] = runs
            population = [
                (run, row)
                for run in runs
                for row in search_llm_calls(run, version)
            ]
            chosen = rng.sample(population, min(SAMPLE_SIZE, len(population)))
            for run, row in chosen:
                prompt = (
                    reconstruct_v9_prompt(run, row)
                    if version == "v9"
                    else str(row["prompt"])
                )
                sampled.append(
                    {
                        "task": task,
                        "version": version,
                        "run": run.name,
                        "sample_order": int(row["sample_order"]),
                        "logged_prompt_tokens_raw": int(row["prompt_tokens"]),
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "prompt": prompt,
                        "blocks": prompt_blocks(version, prompt),
                    }
                )
    return sampled, inventory


def summarize_samples(sampled: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task in TASKS:
        result[task] = {}
        for version in ("v9", "v95"):
            rows = [
                item
                for item in sampled
                if item["task"] == task and item["version"] == version
            ]
            components = sorted(
                {key for row in rows for key in row["component_tokens"]}
            )
            result[task][version] = {
                "n": len(rows),
                "mean_logged_prompt_tokens_raw": mean(
                    [float(row["logged_prompt_tokens_raw"]) for row in rows]
                ),
                "mean_recounted_prompt_tokens_raw": mean(
                    [float(row["recounted_raw_tokens"]) for row in rows]
                ),
                "mean_recounted_chat_tokens": mean(
                    [float(row["recounted_chat_tokens"]) for row in rows]
                ),
                "mean_abs_reconstruction_token_error": mean(
                    [
                        abs(
                            float(row["logged_prompt_tokens_raw"])
                            - float(row["recounted_raw_tokens"])
                        )
                        for row in rows
                    ]
                ),
                "mean_component_tokens": {
                    component: mean(
                        [float(row["component_tokens"].get(component, 0)) for row in rows]
                    )
                    for component in components
                },
            }
    return result


def write_sample_csv(path: Path, sampled: list[dict[str, Any]]) -> None:
    component_names = sorted(
        {name for item in sampled for name in item["component_tokens"]}
    )
    component_columns = {
        name: f"component_{name}_tokens" for name in component_names
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "version",
                "run",
                "sample_order",
                "prompt_sha256",
                "logged_prompt_tokens_raw",
                "recounted_raw_tokens",
                "recounted_chat_tokens",
                *component_columns.values(),
            ],
        )
        writer.writeheader()
        for item in sampled:
            writer.writerow(
                {
                    key: item.get(key)
                    for key in (
                        "task",
                        "version",
                        "run",
                        "sample_order",
                        "prompt_sha256",
                        "logged_prompt_tokens_raw",
                        "recounted_raw_tokens",
                        "recounted_chat_tokens",
                    )
                }
                | {
                    column: item["component_tokens"].get(name, 0)
                    for name, column in component_columns.items()
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--dump-prompts", type=Path)
    args = parser.parse_args()

    sampled, inventory = sample_prompts()
    token_decomposition(sampled, LiveTokenizer())

    aggregate: dict[str, Any] = {}
    for task in TASKS:
        aggregate[task] = {}
        for version in ("v9", "v95"):
            runs = inventory[task][version]
            aggregate[task][version] = {
                "request": aggregate_request_metrics(runs, version),
                "context": aggregate_context_metrics(runs, version),
                "generation": aggregate_generation_metrics(runs, version),
                "run_names": [run.name for run in runs],
            }

    output = {
        "analysis": "TraceAAD V9 vs V9.5 generation interface audit",
        "sample_seed": SAMPLE_SEED,
        "sample_size_per_task_version": SAMPLE_SIZE,
        "scope": "TSP, CVRP, OP; completed formal runs only",
        "aggregate": aggregate,
        "sample_prompt_decomposition": summarize_samples(sampled),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_sample_csv(args.output_csv, sampled)

    if args.dump_prompts is not None:
        args.dump_prompts.mkdir(parents=True, exist_ok=True)
        for item in sampled:
            name = (
                f"{item['version']}_{item['task']}_{item['run']}_"
                f"sample_{item['sample_order']}.txt"
            )
            (args.dump_prompts / name).write_text(item["prompt"], encoding="utf-8")

    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

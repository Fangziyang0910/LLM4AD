from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

from llm4ad.method.shinka_evo import ShinkaEvo, ShinkaEvoProfiler
from llm4ad.task.optimization.cvrp_aco import CVRPACOEvaluation
from llm4ad.task.optimization.generated_data_config import (
    get_generated_task_kwargs,
)
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.task.optimization.op_aco import OPACOEvaluation
from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.tools.env import resolve_llm_api_key
from llm4ad.tools.llm.llm_api_openai import OpenAIAPI

TaskName = Literal["tsp_construct", "cvrp_aco", "op_aco", "online_bin_packing"]
BackendName = Literal["local", "server1", "zhong"]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
TASKS: tuple[TaskName, ...] = (
    "tsp_construct",
    "cvrp_aco",
    "op_aco",
    "online_bin_packing",
)

# Paper Circle Packing table used generations=150. Fair comparison across
# methods in this repo uses a unified evaluation budget of 1000.
PAPER_MAX_SAMPLE_NUMS = 150
PAPER_NUM_GENERATIONS = 150
FAIR_MAX_SAMPLE_NUMS = 1000
FAIR_NUM_GENERATIONS = 1000
PAPER_NUM_ISLANDS = 2
PAPER_ARCHIVE_SIZE = 40
PAPER_ELITE_SELECTION_RATIO = 0.3
PAPER_NUM_ARCHIVE_INSPIRATIONS = 4
PAPER_NUM_TOP_K_INSPIRATIONS = 2
PAPER_MIGRATION_INTERVAL = 10
PAPER_MIGRATION_RATE = 0.0
PAPER_PARENT_SELECTION_STRATEGY = "weighted"
PAPER_PARENT_SELECTION_LAMBDA = 10.0
PAPER_PATCH_TYPES = ("diff", "full", "cross")
PAPER_PATCH_TYPE_PROBS = (0.45, 0.45, 0.1)
PAPER_MAX_PATCH_RESAMPLES = 3
PAPER_META_REC_INTERVAL = 10
PAPER_LLM_UCB_EXPLORATION_COEF = 1.0


@dataclass(frozen=True, slots=True)
class BackendProfile:
    base_url: str
    model: str
    no_proxy: str


BACKENDS: dict[BackendName, BackendProfile] = {
    "local": BackendProfile(
        base_url="http://127.0.0.1:8001/v1",
        model="Qwen3.6-27B",
        no_proxy="127.0.0.1,localhost,::1",
    ),
    "server1": BackendProfile(
        base_url="http://222.201.145.8:8080/v1",
        model="qwen3.6-27b-awq",
        no_proxy="222.201.145.8,localhost,127.0.0.1,::1",
    ),
    "zhong": BackendProfile(
        base_url="http://183.36.243.124:9000/v1",
        model="/home/fzy/models/Qwen3.6-27B-NVFP4",
        no_proxy="183.36.243.124,localhost,127.0.0.1,::1",
    ),
}


@dataclass(frozen=True, slots=True)
class RunSpec:
    task: TaskName
    backend: BackendName
    base_url: str
    model: str
    no_proxy: str
    max_sample_nums: int = FAIR_MAX_SAMPLE_NUMS
    num_generations: int = FAIR_NUM_GENERATIONS
    num_islands: int = PAPER_NUM_ISLANDS
    archive_size: int = PAPER_ARCHIVE_SIZE
    elite_selection_ratio: float = PAPER_ELITE_SELECTION_RATIO
    num_archive_inspirations: int = PAPER_NUM_ARCHIVE_INSPIRATIONS
    num_top_k_inspirations: int = PAPER_NUM_TOP_K_INSPIRATIONS
    migration_interval: int = PAPER_MIGRATION_INTERVAL
    migration_rate: float = PAPER_MIGRATION_RATE
    parent_selection_strategy: str = PAPER_PARENT_SELECTION_STRATEGY
    parent_selection_lambda: float = PAPER_PARENT_SELECTION_LAMBDA
    patch_types: tuple[str, ...] = PAPER_PATCH_TYPES
    patch_type_probs: tuple[float, ...] = PAPER_PATCH_TYPE_PROBS
    max_patch_resamples: int = PAPER_MAX_PATCH_RESAMPLES
    meta_rec_interval: int = PAPER_META_REC_INTERVAL
    llm_ucb_exploration_coef: float = PAPER_LLM_UCB_EXPLORATION_COEF
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = EXPERIMENTS_ROOT

    @property
    def experiment_root(self) -> Path:
        return self.experiments_root / self.task / "shinka_evo"


def make_run_spec(
    *,
    task: TaskName,
    backend: BackendName = "local",
    base_url: str | None = None,
    model: str | None = None,
    no_proxy: str | None = None,
    max_sample_nums: int = FAIR_MAX_SAMPLE_NUMS,
    num_generations: int = FAIR_NUM_GENERATIONS,
    num_islands: int = PAPER_NUM_ISLANDS,
    archive_size: int = PAPER_ARCHIVE_SIZE,
    elite_selection_ratio: float = PAPER_ELITE_SELECTION_RATIO,
    num_archive_inspirations: int = PAPER_NUM_ARCHIVE_INSPIRATIONS,
    num_top_k_inspirations: int = PAPER_NUM_TOP_K_INSPIRATIONS,
    migration_interval: int = PAPER_MIGRATION_INTERVAL,
    migration_rate: float = PAPER_MIGRATION_RATE,
    parent_selection_strategy: str = PAPER_PARENT_SELECTION_STRATEGY,
    parent_selection_lambda: float = PAPER_PARENT_SELECTION_LAMBDA,
    patch_types: tuple[str, ...] = PAPER_PATCH_TYPES,
    patch_type_probs: tuple[float, ...] = PAPER_PATCH_TYPE_PROBS,
    max_patch_resamples: int = PAPER_MAX_PATCH_RESAMPLES,
    meta_rec_interval: int = PAPER_META_REC_INTERVAL,
    llm_ucb_exploration_coef: float = PAPER_LLM_UCB_EXPLORATION_COEF,
    eval_workers: int | None = None,
    output_tokens: int = 16384,
    seed: int = 0,
    repeat: int | None = None,
    run_name: str | None = None,
    experiments_root: Path = EXPERIMENTS_ROOT,
) -> RunSpec:
    profile = BACKENDS[backend]
    spec = RunSpec(
        task=task,
        backend=backend,
        base_url=base_url or profile.base_url,
        model=model or profile.model,
        no_proxy=no_proxy or profile.no_proxy,
        max_sample_nums=max_sample_nums,
        num_generations=num_generations,
        num_islands=num_islands,
        archive_size=archive_size,
        elite_selection_ratio=elite_selection_ratio,
        num_archive_inspirations=num_archive_inspirations,
        num_top_k_inspirations=num_top_k_inspirations,
        migration_interval=migration_interval,
        migration_rate=migration_rate,
        parent_selection_strategy=parent_selection_strategy,
        parent_selection_lambda=parent_selection_lambda,
        patch_types=patch_types,
        patch_type_probs=patch_type_probs,
        max_patch_resamples=max_patch_resamples,
        meta_rec_interval=meta_rec_interval,
        llm_ucb_exploration_coef=llm_ucb_exploration_coef,
        eval_workers=eval_workers,
        output_tokens=output_tokens,
        seed=seed,
        repeat=repeat,
        run_name=run_name,
        experiments_root=experiments_root.resolve(),
    )
    _validate_spec(spec)
    return spec


def _validate_spec(spec: RunSpec) -> None:
    if spec.max_sample_nums <= 0:
        raise ValueError("max_sample_nums must be positive")
    if spec.num_generations <= 0:
        raise ValueError("num_generations must be positive")
    if spec.num_islands <= 0:
        raise ValueError("num_islands must be positive")
    if spec.archive_size <= 0:
        raise ValueError("archive_size must be positive")
    if not 0.0 < spec.elite_selection_ratio <= 1.0:
        raise ValueError("elite_selection_ratio must be in (0, 1]")
    if len(spec.patch_types) != len(spec.patch_type_probs):
        raise ValueError("patch_types and patch_type_probs must have the same length")
    if abs(sum(spec.patch_type_probs) - 1.0) > 1e-6:
        raise ValueError("patch_type_probs must sum to 1")
    if spec.eval_workers is not None and spec.eval_workers <= 0:
        raise ValueError("eval_workers must be positive")
    if spec.output_tokens <= 0:
        raise ValueError("output_tokens must be positive")


def build_task(spec: RunSpec) -> tuple[Any, dict[str, Any]]:
    if spec.task == "tsp_construct":
        kwargs = get_generated_task_kwargs(spec.task, "train")
        return TSPEvaluation(**kwargs), {"split": "train", **kwargs}
    if spec.task == "online_bin_packing":
        kwargs = get_generated_task_kwargs(spec.task, "train")
        return OBPEvaluation(**kwargs), {"split": "train", **kwargs}
    if spec.task == "cvrp_aco":
        kwargs = {
            "split": "train",
            "timeout_seconds": 120,
            "n_ants": 30,
            "n_iterations": 100,
            "aco_seed": 1234,
            "n_workers": spec.eval_workers or 10,
        }
        return CVRPACOEvaluation(**kwargs), kwargs

    kwargs = {
        "split": "train",
        "timeout_seconds": 60,
        "n_ants": 20,
        "n_iterations": 50,
        "aco_seed": 1234,
        "n_workers": spec.eval_workers or 5,
    }
    return OPACOEvaluation(**kwargs), kwargs


def _make_llm(spec: RunSpec) -> OpenAIAPI:
    return OpenAIAPI(
        base_url=spec.base_url,
        api_key=resolve_llm_api_key(base_url=spec.base_url),
        model=spec.model,
        timeout=600,
        max_tokens=spec.output_tokens,
        temperature=1.0,
        enable_thinking=False,
    )


def build_method(spec: RunSpec, log_dir: Path) -> ShinkaEvo:
    os.environ["NO_PROXY"] = spec.no_proxy
    os.environ["no_proxy"] = spec.no_proxy
    random.seed(spec.seed)
    np.random.seed(spec.seed)

    evaluation, _ = build_task(spec)
    llm = _make_llm(spec)
    # Paper uses a separate meta model; with a single Qwen endpoint we reuse it
    # so meta-scratchpad updates remain active.
    meta_llm = _make_llm(spec)
    return ShinkaEvo(
        llm=llm,
        evaluation=evaluation,
        profiler=ShinkaEvoProfiler(
            log_dir=str(log_dir),
            log_style="complex",
            create_random_path=False,
        ),
        max_sample_nums=spec.max_sample_nums,
        num_generations=spec.num_generations,
        num_islands=spec.num_islands,
        archive_size=spec.archive_size,
        patch_types=spec.patch_types,
        patch_type_probs=spec.patch_type_probs,
        parent_selection_strategy=spec.parent_selection_strategy,
        meta_llm=meta_llm,
        novelty_llm=None,
        embedding_fn=None,
        max_patch_resamples=spec.max_patch_resamples,
        elite_selection_ratio=spec.elite_selection_ratio,
        num_archive_inspirations=spec.num_archive_inspirations,
        num_top_k_inspirations=spec.num_top_k_inspirations,
        parent_selection_lambda=spec.parent_selection_lambda,
        migration_interval=spec.migration_interval,
        migration_rate=spec.migration_rate,
        island_elitism=True,
        meta_rec_interval=spec.meta_rec_interval,
        sample_single_meta_rec=True,
        use_text_feedback=False,
        random_seed=spec.seed,
        llm_ucb_exploration_coef=spec.llm_ucb_exploration_coef,
        llm_ucb_epsilon=0.2,
        llm_ucb_auto_decay=0.95,
        max_consecutive_sample_failures=20,
        debug_mode=False,
    )


def resolve_run_dir(spec: RunSpec) -> tuple[Path, str]:
    run_name = spec.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = spec.experiment_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_name


def write_run_config(spec: RunSpec, run_dir: Path, run_name: str) -> None:
    _, task_config = build_task(spec)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "run_name": run_name,
        "task": spec.task,
        "method": "shinka_evo",
        "repeat": spec.repeat,
        "backend": spec.backend,
        "seed": spec.seed,
        "llm": {
            "base_url": spec.base_url,
            "model": spec.model,
            "timeout": 600,
            "max_tokens": spec.output_tokens,
            "temperature": 1.0,
            "enable_thinking": False,
            "no_proxy": spec.no_proxy,
            "api_key_configured": resolve_llm_api_key(base_url=spec.base_url) != "EMPTY",
        },
        "task_eval": task_config,
        "method_params": {
            "max_sample_nums": spec.max_sample_nums,
            "num_generations": spec.num_generations,
            "num_islands": spec.num_islands,
            "archive_size": spec.archive_size,
            "elite_selection_ratio": spec.elite_selection_ratio,
            "num_archive_inspirations": spec.num_archive_inspirations,
            "num_top_k_inspirations": spec.num_top_k_inspirations,
            "migration_interval": spec.migration_interval,
            "migration_rate": spec.migration_rate,
            "parent_selection_strategy": spec.parent_selection_strategy,
            "parent_selection_lambda": spec.parent_selection_lambda,
            "patch_types": list(spec.patch_types),
            "patch_type_probs": list(spec.patch_type_probs),
            "max_patch_resamples": spec.max_patch_resamples,
            "meta_rec_interval": spec.meta_rec_interval,
            "meta_llm_enabled": True,
            "novelty_rejection_enabled": False,
            "embedding_enabled": False,
            "llm_ucb_exploration_coef": spec.llm_ucb_exploration_coef,
            "budget_basis": (
                "Fair comparison budget max_sample_nums=num_generations=1000 "
                "(paper Circle Packing used 150). Other hyperparams follow "
                "Circle Packing table: archive_size=40, islands=2, "
                "inspirations=4/2, migration_rate=0, patch probs [0.45,0.45,0.1], "
                "meta interval=10, UCB exploration=1.0; novelty disabled; "
                "meta LLM reuses the same Qwen endpoint"
            ),
        },
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_experiment(spec: RunSpec) -> Path:
    run_dir, run_name = resolve_run_dir(spec)
    log_dir = run_dir / "logs"
    tmux_log = run_dir / "tmux_run.log"
    write_run_config(spec, run_dir, run_name)
    print(f"run_dir={run_dir}")
    with tmux_log.open("a", encoding="utf-8", buffering=1) as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print(f"run_dir={run_dir}", flush=True)
            print(f"log_dir={log_dir}", flush=True)
            print(f"llm={spec.model} @ {spec.base_url}", flush=True)
            print(
                "shinka_evo="
                f"gens={spec.num_generations}, budget={spec.max_sample_nums}, "
                f"archive={spec.archive_size}, islands={spec.num_islands}, "
                f"insp={spec.num_archive_inspirations}/{spec.num_top_k_inspirations}",
                flush=True,
            )
            build_method(spec, log_dir).run()
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one paper-aligned ShinkaEvolve experiment."
    )
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--max-sample-nums", type=int, default=FAIR_MAX_SAMPLE_NUMS)
    parser.add_argument("--num-generations", type=int, default=FAIR_NUM_GENERATIONS)
    parser.add_argument("--eval-workers", type=int)
    parser.add_argument("--output-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--run-name")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = make_run_spec(
        task=args.task,
        backend=args.backend,
        base_url=args.base_url,
        model=args.model,
        no_proxy=args.no_proxy,
        max_sample_nums=args.max_sample_nums,
        num_generations=args.num_generations,
        eval_workers=args.eval_workers,
        output_tokens=args.output_tokens,
        seed=args.seed,
        repeat=args.repeat,
        run_name=args.run_name,
    )
    run_experiment(spec)


if __name__ == "__main__":
    main()

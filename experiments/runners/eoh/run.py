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

from llm4ad.method.eoh import EoH, EoHProfiler
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

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
TASKS: tuple[TaskName, ...] = (
    "tsp_construct",
    "cvrp_aco",
    "op_aco",
    "online_bin_packing",
)
OPERATORS = ("e1", "e2", "m1", "m2")
PAPER_GENERATIONS = 20
PAPER_STRATEGY_COUNT = 5
FORMAL_BUDGET = 1000


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


def paper_population_size(task: TaskName) -> int:
    return 20 if task == "online_bin_packing" else 10


def paper_query_budget(task: TaskName, generations: int = PAPER_GENERATIONS) -> int:
    del task, generations
    return FORMAL_BUDGET


@dataclass(frozen=True, slots=True)
class RunSpec:
    task: TaskName
    backend: BackendName
    base_url: str
    model: str
    no_proxy: str
    generations: int = PAPER_GENERATIONS
    budget: int | None = None
    pop_size: int | None = None
    parents: int = 5
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = EXPERIMENTS_ROOT

    @property
    def effective_pop_size(self) -> int:
        return self.pop_size or paper_population_size(self.task)

    @property
    def effective_budget(self) -> int:
        if self.budget is not None:
            return self.budget
        return FORMAL_BUDGET

    @property
    def experiment_root(self) -> Path:
        return self.experiments_root / self.task / "eoh"


def make_run_spec(
    *,
    task: TaskName,
    backend: BackendName = "local",
    base_url: str | None = None,
    model: str | None = None,
    no_proxy: str | None = None,
    generations: int = PAPER_GENERATIONS,
    budget: int | None = None,
    pop_size: int | None = None,
    parents: int = 5,
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
        generations=generations,
        budget=budget,
        pop_size=pop_size,
        parents=parents,
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
    if spec.generations <= 0:
        raise ValueError("generations must be positive")
    if spec.effective_budget <= 0:
        raise ValueError("budget must be positive")
    if spec.effective_pop_size <= 0:
        raise ValueError("pop_size must be positive")
    if not 2 <= spec.parents <= spec.effective_pop_size:
        raise ValueError("parents must be between 2 and pop_size")
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


def build_method(spec: RunSpec, log_dir: Path) -> EoH:
    os.environ["NO_PROXY"] = spec.no_proxy
    os.environ["no_proxy"] = spec.no_proxy
    random.seed(spec.seed)
    np.random.seed(spec.seed)

    evaluation, _ = build_task(spec)
    llm = OpenAIAPI(
        base_url=spec.base_url,
        api_key=resolve_llm_api_key(base_url=spec.base_url),
        model=spec.model,
        timeout=600,
        max_tokens=spec.output_tokens,
        temperature=1.0,
        enable_thinking=False,
    )
    return EoH(
        llm=llm,
        evaluation=evaluation,
        profiler=EoHProfiler(
            log_dir=str(log_dir),
            log_style="complex",
            create_random_path=False,
        ),
        max_generations=spec.generations,
        max_sample_nums=spec.effective_budget,
        pop_size=spec.effective_pop_size,
        selection_num=spec.parents,
        operators=list(OPERATORS),
        operator_weights=[1.0] * len(OPERATORS),
        use_m3_operator=False,
        num_samplers=1,
        num_evaluators=1,
        max_consecutive_sample_failures=20,
        multi_thread_or_process_eval="thread",
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
        "method": "eoh",
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
            "paper_generations": spec.generations,
            "max_sample_nums": spec.effective_budget,
            "population_size": spec.effective_pop_size,
            "initialization_samples": 2 * spec.effective_pop_size,
            "selection_num": spec.parents,
            "operators": list(OPERATORS),
            "operator_weights": [1.0] * len(OPERATORS),
            "m3_enabled": False,
            "num_samplers": 1,
            "num_evaluators": 1,
            "budget_basis": (
                "all formal task comparisons use a fixed 1000-evaluation budget"
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
                "eoh="
                f"pop={spec.effective_pop_size}, parents={spec.parents}, "
                f"budget={spec.effective_budget}, operators={','.join(OPERATORS)}",
                flush=True,
            )
            build_method(spec, log_dir).run()
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one paper-aligned EoH experiment.")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--generations", type=int, default=PAPER_GENERATIONS)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--pop-size", type=int)
    parser.add_argument("--parents", type=int, default=5)
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
        generations=args.generations,
        budget=args.budget,
        pop_size=args.pop_size,
        parents=args.parents,
        eval_workers=args.eval_workers,
        output_tokens=args.output_tokens,
        seed=args.seed,
        repeat=args.repeat,
        run_name=args.run_name,
    )
    run_experiment(spec)


if __name__ == "__main__":
    main()

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

from llm4ad.method.reevo import ReEvo, ReEvoProfiler
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

# Paper table / original cfg/config.yaml used max_fe=100 for sample-efficiency
# claims. Fair comparison across methods in this repo uses a unified budget.
PAPER_MAX_SAMPLE_NUMS = 100
FAIR_MAX_SAMPLE_NUMS = 1000
PAPER_POP_SIZE = 10
PAPER_INIT_POP_SIZE = 30
PAPER_MUTATION_RATE = 0.5


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
    pop_size: int = PAPER_POP_SIZE
    init_pop_size: int = PAPER_INIT_POP_SIZE
    mutation_rate: float = PAPER_MUTATION_RATE
    eval_workers: int | None = None
    output_tokens: int = 16384
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    experiments_root: Path = EXPERIMENTS_ROOT

    @property
    def experiment_root(self) -> Path:
        return self.experiments_root / self.task / "reevo"


def make_run_spec(
    *,
    task: TaskName,
    backend: BackendName = "local",
    base_url: str | None = None,
    model: str | None = None,
    no_proxy: str | None = None,
    max_sample_nums: int = FAIR_MAX_SAMPLE_NUMS,
    pop_size: int = PAPER_POP_SIZE,
    init_pop_size: int = PAPER_INIT_POP_SIZE,
    mutation_rate: float = PAPER_MUTATION_RATE,
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
        pop_size=pop_size,
        init_pop_size=init_pop_size,
        mutation_rate=mutation_rate,
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
    if spec.pop_size <= 0:
        raise ValueError("pop_size must be positive")
    if spec.init_pop_size <= 0:
        raise ValueError("init_pop_size must be positive")
    if not 0.0 < spec.mutation_rate <= 1.0:
        raise ValueError("mutation_rate must be in (0, 1]")
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


def build_method(spec: RunSpec, log_dir: Path) -> ReEvo:
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
    return ReEvo(
        llm=llm,
        evaluation=evaluation,
        profiler=ReEvoProfiler(
            log_dir=str(log_dir),
            log_style="complex",
            create_random_path=False,
        ),
        max_sample_nums=spec.max_sample_nums,
        pop_size=spec.pop_size,
        init_pop_size=spec.init_pop_size,
        mutation_rate=spec.mutation_rate,
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
        "method": "reevo",
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
            "population_size": spec.pop_size,
            "init_pop_size": spec.init_pop_size,
            "mutation_rate": spec.mutation_rate,
            "num_samplers": 1,
            "num_evaluators": 1,
            "init_temperature_offset": 0.3,
            "budget_basis": (
                "Fair comparison budget max_sample_nums=1000 (paper default "
                "max_fe=100 preserved as PAPER_MAX_SAMPLE_NUMS); other "
                "hyperparams follow paper/original cfg: pop_size=10, "
                "init_pop_size=30, mutation_rate=0.5, temperature=1; "
                "initialization uses temperature+0.3"
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
                "reevo="
                f"pop={spec.pop_size}, init={spec.init_pop_size}, "
                f"budget={spec.max_sample_nums}, mutation_rate={spec.mutation_rate}",
                flush=True,
            )
            build_method(spec, log_dir).run()
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one paper-aligned ReEvo experiment.")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--max-sample-nums", type=int, default=FAIR_MAX_SAMPLE_NUMS)
    parser.add_argument("--pop-size", type=int, default=PAPER_POP_SIZE)
    parser.add_argument("--init-pop-size", type=int, default=PAPER_INIT_POP_SIZE)
    parser.add_argument("--mutation-rate", type=float, default=PAPER_MUTATION_RATE)
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
        pop_size=args.pop_size,
        init_pop_size=args.init_pop_size,
        mutation_rate=args.mutation_rate,
        eval_workers=args.eval_workers,
        output_tokens=args.output_tokens,
        seed=args.seed,
        repeat=args.repeat,
        run_name=args.run_name,
    )
    run_experiment(spec)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import contextlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from llm4ad.method.traceaad_artifacts import TraceAADArtifacts
from llm4ad.method.traceaad_v4 import (
    TraceAADV4,
)
from llm4ad.method.traceaad_v4 import (
    ValueWeights as V4ValueWeights,
)
from llm4ad.method.traceaad_v5 import (
    TraceAADV5,
)
from llm4ad.method.traceaad_v5 import (
    ValueWeights as V5ValueWeights,
)
from llm4ad.method.traceaad_v6 import (
    CHECKPOINT_VERSION as V6_CHECKPOINT_VERSION,
)
from llm4ad.method.traceaad_v6 import (
    PROTOCOL_ID as V6_PROTOCOL_ID,
)
from llm4ad.method.traceaad_v6 import (
    TraceAADV6,
)
from llm4ad.method.traceaad_v6 import (
    ValueWeights as V6ValueWeights,
)
from llm4ad.method.traceaad_v6.operators import DEFAULT_OPERATORS as V6_OPERATORS
from llm4ad.method.traceaad_v7 import (
    CHECKPOINT_VERSION as V7_CHECKPOINT_VERSION,
)
from llm4ad.method.traceaad_v7 import (
    PROTOCOL_ID as V7_PROTOCOL_ID,
)
from llm4ad.method.traceaad_v7 import (
    TraceAADV7,
)
from llm4ad.method.traceaad_v7 import (
    ValueWeights as V7ValueWeights,
)
from llm4ad.method.traceaad_v7.operators import DEFAULT_OPERATORS as V7_OPERATORS
from llm4ad.method.traceaad_v7.prompt import (
    ACTION_OUTPUT_MODE as V7_ACTION_OUTPUT_MODE,
)
from llm4ad.method.traceaad_v8 import (
    CHECKPOINT_VERSION as V8_CHECKPOINT_VERSION,
)
from llm4ad.method.traceaad_v8 import (
    PROTOCOL_ID as V8_PROTOCOL_ID,
)
from llm4ad.method.traceaad_v8 import TraceAADV8
from llm4ad.method.traceaad_v8.operators import DEFAULT_OPERATORS as V8_OPERATORS
from llm4ad.method.traceaad_v8_3 import (
    DEFAULT_OPERATORS as V83_OPERATORS,
    PROTOCOL_ID as V83_PROTOCOL_ID,
    SearchWeights as V83SearchWeights,
    TraceAADV8_3,
)
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
VersionName = Literal["v4", "v5", "v6", "v7", "v8", "v8_3"]
BackendName = Literal["local", "server1", "zhong"]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
TASKS: tuple[TaskName, ...] = (
    "tsp_construct",
    "cvrp_aco",
    "op_aco",
    "online_bin_packing",
)
VERSIONS: tuple[VersionName, ...] = ("v4", "v5", "v6", "v7", "v8", "v8_3")
V6_OPERATOR_NAMES = [str(operator_type.name) for operator_type in V6_OPERATORS]
V7_OPERATOR_NAMES = [str(operator.name) for operator in V7_OPERATORS]
V8_OPERATOR_NAMES = [str(operator_type.name) for operator_type in V8_OPERATORS]
V83_OPERATOR_NAMES = [str(operator.name) for operator in V83_OPERATORS]


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
    version: VersionName
    backend: BackendName
    base_url: str
    model: str
    no_proxy: str
    n_init: int
    budget: int = 1000
    eval_workers: int | None = None
    output_tokens: int | None = None
    action_max_tokens: int = 1024
    context_token_limit: int = 24576
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    resume_from: Path | None = None
    experiments_root: Path = EXPERIMENTS_ROOT

    @property
    def method_name(self) -> str:
        return f"traceaad_{self.version}"

    @property
    def experiment_version(self) -> str:
        return f"version{self.version.removeprefix('v')}"

    @property
    def experiment_root(self) -> Path:
        return self.experiments_root / self.task / self.method_name

    @property
    def llm_output_tokens(self) -> int:
        if self.output_tokens is not None:
            return self.output_tokens
        return 16384 if self.version in {"v4", "v7"} else 8192


def make_run_spec(
    *,
    task: TaskName,
    version: VersionName,
    backend: BackendName = "local",
    base_url: str | None = None,
    model: str | None = None,
    no_proxy: str | None = None,
    budget: int = 1000,
    n_init: int | None = None,
    eval_workers: int | None = None,
    output_tokens: int | None = None,
    action_max_tokens: int = 1024,
    context_token_limit: int = 24576,
    seed: int = 0,
    repeat: int | None = None,
    run_name: str | None = None,
    resume_from: Path | None = None,
    experiments_root: Path = EXPERIMENTS_ROOT,
) -> RunSpec:
    profile = BACKENDS[backend]
    spec = RunSpec(
        task=task,
        version=version,
        backend=backend,
        base_url=base_url or profile.base_url,
        model=model or profile.model,
        no_proxy=no_proxy or profile.no_proxy,
        budget=budget,
        n_init=(10 if version in {"v8", "v8_3"} else 30) if n_init is None else n_init,
        eval_workers=eval_workers,
        output_tokens=output_tokens,
        action_max_tokens=action_max_tokens,
        context_token_limit=context_token_limit,
        seed=seed,
        repeat=repeat,
        run_name=run_name,
        resume_from=None if resume_from is None else resume_from.resolve(),
        experiments_root=experiments_root.resolve(),
    )
    _validate_spec(spec)
    return spec


def _validate_spec(spec: RunSpec) -> None:
    if spec.budget <= 0:
        raise ValueError("budget must be positive")
    if spec.n_init <= 0:
        raise ValueError("n_init must be positive")
    if spec.eval_workers is not None and spec.eval_workers <= 0:
        raise ValueError("eval_workers must be positive")
    if spec.llm_output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    if spec.action_max_tokens <= 0:
        raise ValueError("action_max_tokens must be positive")
    if spec.context_token_limit <= 0:
        raise ValueError("context_token_limit must be positive")
    if spec.resume_from is not None and spec.run_name is not None:
        raise ValueError("run_name cannot be combined with resume_from")


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


def build_method(
    spec: RunSpec,
    run_dir: Path,
    resume_from: Path | None = None,
) -> TraceAADV4 | TraceAADV5 | TraceAADV6 | TraceAADV7 | TraceAADV8 | TraceAADV8_3:
    os.environ["NO_PROXY"] = spec.no_proxy
    os.environ["no_proxy"] = spec.no_proxy
    evaluation, _ = build_task(spec)
    llm = OpenAIAPI(
        base_url=spec.base_url,
        api_key=resolve_llm_api_key(base_url=spec.base_url),
        model=spec.model,
        timeout=600,
        max_tokens=spec.llm_output_tokens,
        temperature=1.0,
        enable_thinking=False,
    )
    artifacts = TraceAADArtifacts(run_dir=run_dir)
    if spec.version == "v8_3":
        search_weights = V83SearchWeights()
        return TraceAADV8_3(
            llm=llm,
            evaluation=evaluation,
            profiler=artifacts,
            max_sample_nums=spec.budget,
            n_init=spec.n_init,
            search_weights=search_weights,
            reference_temperature=1.0,
            context_token_limit=spec.context_token_limit,
            retry_count=3,
            max_consecutive_sample_failures=20,
            random_seed=spec.seed,
        )
    common = {
        "llm": llm,
        "evaluation": evaluation,
        "max_sample_nums": spec.budget,
        "n_init": spec.n_init,
        "max_consecutive_sample_failures": 20,
        "max_stalled_iterations": 20,
        "checkpoint_interval": 10,
        "resume_from": resume_from,
        "checkpoint_dir": run_dir / "checkpoints",
    }
    if spec.version == "v8":
        return TraceAADV8(
            profiler=artifacts,
            ancestor_history_limit=8,
            direct_child_limit=8,
            direct_child_top_count=4,
            reference_temperature=0.2,
            exploration_constant=0.1,
            expansion_prior_weight=1.0,
            offspring_per_iteration=2,
            code_max_tokens=spec.llm_output_tokens,
            context_token_limit=spec.context_token_limit,
            random_seed=spec.seed,
            **common,
        )
    population_common = {
        "actions_per_iteration": 2,
        "max_trajectory_length": 8,
        "max_active_trajectories": 30,
        "softmax_temperature": 0.2,
    }
    if spec.version == "v4":
        return TraceAADV4(
            profiler=artifacts,
            value_weights=V4ValueWeights(),
            **population_common,
            **common,
        )
    if spec.version == "v5":
        return TraceAADV5(
            profiler=artifacts,
            value_weights=V5ValueWeights(),
            elite_count=3,
            action_max_tokens=spec.action_max_tokens,
            random_seed=spec.seed,
            **population_common,
            **common,
        )
    if spec.version == "v6":
        return TraceAADV6(
            profiler=artifacts,
            value_weights=V6ValueWeights(),
            action_max_tokens=spec.action_max_tokens,
            code_max_tokens=spec.llm_output_tokens,
            context_token_limit=spec.context_token_limit,
            random_seed=spec.seed,
            **population_common,
            **common,
        )
    return TraceAADV7(
        profiler=artifacts,
        value_weights=V7ValueWeights(),
        elite_count=3,
        action_max_tokens=spec.action_max_tokens,
        code_max_tokens=spec.llm_output_tokens,
        context_token_limit=spec.context_token_limit,
        random_seed=spec.seed,
        **population_common,
        **common,
    )


def resolve_run_dir(spec: RunSpec) -> tuple[Path, str, bool]:
    if spec.resume_from is not None:
        run_dir = spec.resume_from
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")
        _validate_resume_config(spec, run_dir)
        return run_dir, run_dir.name, True

    run_name = spec.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = spec.experiment_root / spec.experiment_version / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, run_name, False


def _validate_resume_config(spec: RunSpec, run_dir: Path) -> None:
    path = run_dir / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"resume config does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {"task": spec.task, "method": spec.method_name}
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"resume config mismatch: expected {expected}, found {actual}")
    if spec.version not in {"v6", "v7", "v8"}:
        return
    _, task_kwargs = build_task(spec)
    normalized_task_kwargs = json.loads(json.dumps(task_kwargs, sort_keys=True))
    if spec.version == "v6":
        protocol_id = V6_PROTOCOL_ID
        checkpoint_version = V6_CHECKPOINT_VERSION
        operator_names = V6_OPERATOR_NAMES
        weights = V6ValueWeights()
    elif spec.version == "v7":
        protocol_id = V7_PROTOCOL_ID
        checkpoint_version = V7_CHECKPOINT_VERSION
        operator_names = V7_OPERATOR_NAMES
        weights = V7ValueWeights()
    else:
        protocol_id = V8_PROTOCOL_ID
        checkpoint_version = V8_CHECKPOINT_VERSION
        operator_names = V8_OPERATOR_NAMES
        weights = None
    if spec.version == "v8":
        expected_method_params = {
            "protocol_id": protocol_id,
            "checkpoint_schema_version": checkpoint_version,
            "max_sample_nums": spec.budget,
            "n_init": spec.n_init,
            "offspring_per_iteration": 2,
            "generation_protocol": "direct_code",
            "quality_normalization": "global_midrank_percentile",
            "expansion_policy": "adaptive_new_child_uct",
            "expansion_reward": "batch_subtree_best_midrank",
            "failed_expansion_reward": 0.0,
            "root_expansion": False,
            "ancestor_history_limit": 8,
            "direct_child_limit": 8,
            "direct_child_top_count": 4,
            "reference_temperature": 0.2,
            "exploration_constant": 0.1,
            "expansion_prior_weight": 1.0,
            "maximize": True,
            "operators": operator_names,
            "max_consecutive_sample_failures": 20,
            "max_stalled_iterations": 20,
            "checkpoint_interval": 10,
            "code_max_tokens": spec.llm_output_tokens,
            "context_token_limit": spec.context_token_limit,
            "random_seed": spec.seed,
        }
    else:
        expected_method_params = {
            "protocol_id": protocol_id,
            "checkpoint_schema_version": checkpoint_version,
            "max_sample_nums": spec.budget,
            "n_init": spec.n_init,
            "actions_per_iteration": 2,
            "max_trajectory_length": 8,
            "max_active_trajectories": 30,
            "softmax_temperature": 0.2,
            "maximize": True,
            "operators": operator_names,
            "max_consecutive_sample_failures": 20,
            "max_stalled_iterations": 20,
            "checkpoint_interval": 10,
            "value_weights": asdict(weights),
            "action_max_tokens": spec.action_max_tokens,
            "code_max_tokens": spec.llm_output_tokens,
            "context_token_limit": spec.context_token_limit,
            "random_seed": spec.seed,
        }
    expected_protocol = {
        "backend": spec.backend,
        "task_eval": normalized_task_kwargs,
        "llm": {
            "base_url": spec.base_url,
            "model": spec.model,
            "max_tokens": spec.llm_output_tokens,
            "no_proxy": spec.no_proxy,
        },
        "method_params": expected_method_params,
    }
    if spec.version == "v7":
        expected_protocol["method_params"]["elite_count"] = 3
        expected_protocol["method_params"]["action_output_mode"] = V7_ACTION_OUTPUT_MODE
    actual_protocol = {
        "backend": payload.get("backend"),
        "task_eval": payload.get("task_eval"),
        "llm": {
            key: payload.get("llm", {}).get(key) for key in expected_protocol["llm"]
        },
        "method_params": {
            key: payload.get("method_params", {}).get(key)
            for key in expected_protocol["method_params"]
        },
    }
    if actual_protocol != expected_protocol:
        raise ValueError(
            f"resume config mismatch for TraceAAD {spec.version.upper()}; "
            "use the original model, "
            "evaluation, budget, seed, and context settings"
        )


def checkpoint_source(spec: RunSpec, run_dir: Path) -> Path:
    """Return the checkpoint path used for resume.

    Canonical location is ``run_dir/checkpoints/latest.json``. V4 also accepts
    legacy locations when the canonical file is missing.
    """
    canonical = run_dir / "checkpoints" / "latest.json"
    if canonical.is_file() or spec.version != "v4":
        return canonical
    from llm4ad.method.traceaad_v4.checkpoint import find_latest_checkpoint

    try:
        return find_latest_checkpoint(run_dir)
    except FileNotFoundError:
        return canonical


def write_run_config(spec: RunSpec, run_dir: Path, run_name: str) -> None:
    _, task_kwargs = build_task(spec)
    if spec.version == "v4":
        weights = V4ValueWeights()
    elif spec.version == "v5":
        weights = V5ValueWeights()
    elif spec.version == "v6":
        weights = V6ValueWeights()
    elif spec.version == "v7":
        weights = V7ValueWeights()
    else:
        weights = None
    method_params: dict[str, Any]
    if spec.version == "v8_3":
        search_weights = V83SearchWeights()
        method_params = {
            "protocol_id": V83_PROTOCOL_ID,
            "maximize": True,
            "max_sample_nums": spec.budget,
            "n_init": spec.n_init,
            "generation_protocol": "call1_design_idea_code_then_call2_description",
            "quality_normalization": "global_midrank_percentile",
            "trajectory_prior": "endpoint_path_best_and_recent_trend",
            "credit_assignment": "direct_reward_mean_on_selected_route",
            "expansion_policy": "progressive_widening_then_softmax_ucb",
            "expansion_reward": "frozen_child_quality_plus_positive_parent_gain",
            "failed_expansion_reward": 0.0,
            "root_expansion": True,
            "search_weights": asdict(search_weights),
            "reference_temperature": 1.0,
            "max_consecutive_sample_failures": 20,
            "retry_count": 3,
            "context_token_limit": spec.context_token_limit,
            "random_seed": spec.seed,
            "operators": V83_OPERATOR_NAMES,
        }
    elif spec.version == "v8":
        method_params = {
            "max_sample_nums": spec.budget,
            "n_init": spec.n_init,
            "offspring_per_iteration": 2,
            "generation_protocol": "direct_code",
            "quality_normalization": "global_midrank_percentile",
            "expansion_policy": "adaptive_new_child_uct",
            "expansion_reward": "batch_subtree_best_midrank",
            "failed_expansion_reward": 0.0,
            "root_expansion": False,
            "ancestor_history_limit": 8,
            "direct_child_limit": 8,
            "direct_child_top_count": 4,
            "reference_temperature": 0.2,
            "exploration_constant": 0.1,
            "expansion_prior_weight": 1.0,
            "max_consecutive_sample_failures": 20,
            "max_stalled_iterations": 20,
            "checkpoint_interval": 10,
        }
    else:
        method_params = {
            "max_sample_nums": spec.budget,
            "n_init": spec.n_init,
            "actions_per_iteration": 2,
            "max_trajectory_length": 8,
            "max_active_trajectories": 30,
            "softmax_temperature": 0.2,
            "max_consecutive_sample_failures": 20,
            "max_stalled_iterations": 20,
            "checkpoint_interval": 10,
            "value_weights": asdict(weights),
        }
    if spec.version == "v5":
        method_params.update(
            {
                "elite_count": 3,
                "action_max_tokens": spec.action_max_tokens,
                "random_seed": spec.seed,
            }
        )
    if spec.version == "v6":
        method_params.update(
            {
                "protocol_id": V6_PROTOCOL_ID,
                "checkpoint_schema_version": V6_CHECKPOINT_VERSION,
                "maximize": True,
                "operators": V6_OPERATOR_NAMES,
                "action_max_tokens": spec.action_max_tokens,
                "code_max_tokens": spec.llm_output_tokens,
                "context_token_limit": spec.context_token_limit,
                "random_seed": spec.seed,
            }
        )
    if spec.version == "v7":
        method_params.update(
            {
                "protocol_id": V7_PROTOCOL_ID,
                "checkpoint_schema_version": V7_CHECKPOINT_VERSION,
                "maximize": True,
                "operators": V7_OPERATOR_NAMES,
                "elite_count": 3,
                "action_max_tokens": spec.action_max_tokens,
                "action_output_mode": V7_ACTION_OUTPUT_MODE,
                "code_max_tokens": spec.llm_output_tokens,
                "context_token_limit": spec.context_token_limit,
                "random_seed": spec.seed,
            }
        )
    if spec.version == "v8":
        method_params.update(
            {
                "protocol_id": V8_PROTOCOL_ID,
                "checkpoint_schema_version": V8_CHECKPOINT_VERSION,
                "maximize": True,
                "operators": V8_OPERATOR_NAMES,
                "code_max_tokens": spec.llm_output_tokens,
                "context_token_limit": spec.context_token_limit,
                "random_seed": spec.seed,
            }
        )
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "task": spec.task,
        "method": spec.method_name,
        "experiment_version": spec.experiment_version,
        "timestamp": run_name,
        "repeat": spec.repeat,
        "backend": spec.backend,
        "llm": {
            "base_url": spec.base_url,
            "model": spec.model,
            "timeout": 600,
            "max_tokens": spec.llm_output_tokens,
            "temperature": 1.0,
            "enable_thinking": False,
            "api_key_configured": (
                resolve_llm_api_key(base_url=spec.base_url) != "EMPTY"
            ),
            "no_proxy": spec.no_proxy,
        },
        "task_eval": task_kwargs,
        "method_params": method_params,
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_experiment(spec: RunSpec) -> Path:
    run_dir, run_name, resumed = resolve_run_dir(spec)
    if not resumed:
        write_run_config(spec, run_dir, run_name)

    print(f"run_dir={run_dir}")
    resume_source = checkpoint_source(spec, run_dir) if resumed else None
    with (run_dir / "tmux_run.log").open(
        "a", encoding="utf-8", buffering=1
    ) as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            print(f"run_dir={run_dir}", flush=True)
            build_method(spec, run_dir, resume_source).run()
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one TraceAAD experiment with an explicit task and version."
    )
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--version", required=True, choices=VERSIONS)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--n-init", type=int)
    parser.add_argument("--eval-workers", type=int)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--action-max-tokens", type=int, default=1024)
    parser.add_argument("--context-token-limit", type=int, default=24576)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from", type=Path)
    return parser


def spec_from_args(args: argparse.Namespace) -> RunSpec:
    return make_run_spec(
        task=args.task,
        version=args.version,
        backend=args.backend,
        base_url=args.base_url,
        model=args.model,
        no_proxy=args.no_proxy,
        budget=args.budget,
        n_init=args.n_init,
        eval_workers=args.eval_workers,
        output_tokens=args.output_tokens,
        action_max_tokens=args.action_max_tokens,
        context_token_limit=args.context_token_limit,
        seed=args.seed,
        repeat=args.repeat,
        run_name=args.run_name,
        resume_from=args.resume_from,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_experiment(spec_from_args(args))


if __name__ == "__main__":
    main()

"""Run one TraceAAD experiment with an explicit task and version."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from llm4ad.method.traceaad_artifacts import TraceAADArtifacts
from llm4ad.method.traceaad_v4 import TraceAADV4, ValueWeights as V4ValueWeights
from llm4ad.method.traceaad_v5 import TraceAADV5, ValueWeights as V5ValueWeights
from llm4ad.method.traceaad_v8 import (
    CHECKPOINT_VERSION as V8_CHECKPOINT_VERSION,
    PROTOCOL_ID as V8_PROTOCOL_ID,
    TraceAADV8,
)
from llm4ad.method.traceaad_v8.operators import DEFAULT_OPERATORS as V8_OPERATORS
from llm4ad.method.traceaad_v9 import (
    CHECKPOINT_VERSION as V9_CHECKPOINT_VERSION,
    PROTOCOL_ID as V9_PROTOCOL_ID,
    TraceAADV9,
)
from llm4ad.method.traceaad_v9.operators import DEFAULT_OPERATORS as V9_OPERATORS
from llm4ad.method.traceaad_v9_1 import (
    CHECKPOINT_VERSION as V91_CHECKPOINT_VERSION,
    PROTOCOL_ID as V91_PROTOCOL_ID,
    TraceAADV91,
)
from llm4ad.method.traceaad_v9_1.operators import DEFAULT_OPERATORS as V91_OPERATORS

from .._common import (
    BACKENDS,
    EXPERIMENTS_ROOT,
    TASKS,
    TaskName,
    build_llm_client,
    build_task,
    llm_payload,
    resolve_backend,
    resolve_run_dir as resolve_run_dir_file,
    run_in_tmux_log,
    write_run_config as write_run_config_file,
)

VersionName = Literal["v4", "v5", "v8", "v9", "v9_1"]

VERSIONS: tuple[VersionName, ...] = ("v4", "v5", "v8", "v9", "v9_1")
V8_OPERATOR_NAMES = [str(operator_type.name) for operator_type in V8_OPERATORS]
V9_OPERATOR_NAMES = [str(operator_type.name) for operator_type in V9_OPERATORS]
V91_OPERATOR_NAMES = [str(operator_type.name) for operator_type in V91_OPERATORS]


@dataclass(frozen=True, slots=True)
class RunSpec:
    task: TaskName
    version: VersionName
    backend: str
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
    def experiment_root(self) -> Path:
        return self.experiments_root / self.task / self.method_name

    @property
    def llm_output_tokens(self) -> int:
        if self.output_tokens is not None:
            return self.output_tokens
        return 16384 if self.version == "v4" else 8192


def make_run_spec(
    *,
    task: TaskName,
    version: VersionName,
    backend: str = "local",
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
    profile = resolve_backend(backend, base_url, model, no_proxy)
    spec = RunSpec(
        task=task,
        version=version,
        backend=backend,
        base_url=profile.base_url,
        model=profile.model,
        no_proxy=profile.no_proxy,
        budget=budget,
        n_init=(4 if version == "v9_1" else 10 if version in {"v8", "v9"} else 30)
        if n_init is None
        else n_init,
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
    return spec


def build_method(
    spec: RunSpec,
    run_dir: Path,
    resume_from: Path | None = None,
):
    evaluation, _ = build_task(spec.task, spec.eval_workers)
    llm = build_llm_client(
        base_url=spec.base_url,
        model=spec.model,
        no_proxy=spec.no_proxy,
        max_tokens=spec.llm_output_tokens,
        temperature=1.0,
    )
    artifacts = TraceAADArtifacts(run_dir=run_dir)
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
    if spec.version in {"v8", "v9", "v9_1"}:
        method_type = (
            TraceAADV8
            if spec.version == "v8"
            else TraceAADV9
            if spec.version == "v9"
            else TraceAADV91
        )
        tree_kwargs = {
            "ancestor_history_limit": 8,
            "direct_child_limit": 8,
            "direct_child_top_count": 4,
            "reference_temperature": 0.2,
            "exploration_constant": 0.1,
            "code_max_tokens": spec.llm_output_tokens,
            "context_token_limit": spec.context_token_limit,
            "random_seed": spec.seed,
            "offspring_per_iteration": 1 if spec.version == "v9_1" else 2,
        }
        if spec.version == "v9_1":
            tree_kwargs["alpha"] = 0.5
        else:
            tree_kwargs["expansion_prior_weight"] = 1.0
        return method_type(
            profiler=artifacts,
            **tree_kwargs,
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
    return TraceAADV5(
        profiler=artifacts,
        value_weights=V5ValueWeights(),
        elite_count=3,
        action_max_tokens=spec.action_max_tokens,
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
    run_dir, run_name = resolve_run_dir_file(spec.experiment_root, spec.run_name)
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
    if spec.version not in {"v8", "v9", "v9_1"}:
        return
    _, task_kwargs = build_task(spec.task, spec.eval_workers)
    normalized_task_kwargs = json.loads(json.dumps(task_kwargs, sort_keys=True))
    if spec.version == "v8":
        protocol_id = V8_PROTOCOL_ID
        checkpoint_version = V8_CHECKPOINT_VERSION
        operator_names = V8_OPERATOR_NAMES
    elif spec.version == "v9":
        protocol_id = V9_PROTOCOL_ID
        checkpoint_version = V9_CHECKPOINT_VERSION
        operator_names = V9_OPERATOR_NAMES
    else:
        protocol_id = V91_PROTOCOL_ID
        checkpoint_version = V91_CHECKPOINT_VERSION
        operator_names = V91_OPERATOR_NAMES
    expected_method_params = {
        "protocol_id": protocol_id,
        "checkpoint_schema_version": checkpoint_version,
        "max_sample_nums": spec.budget,
        "n_init": spec.n_init,
        "offspring_per_iteration": 1 if spec.version == "v9_1" else 2,
        "generation_protocol": "direct_code",
        "quality_normalization": (
            "mcts_minmax_q"
            if spec.version == "v9_1"
            else "global_midrank_percentile"
        ),
        "expansion_policy": (
            "progressive_widening_uct"
            if spec.version == "v9_1"
            else "adaptive_new_child_uct"
        ),
        "expansion_reward": (
            "child_continuation_value"
            if spec.version == "v9_1"
            else "batch_subtree_best_midrank"
        ),
        **(
            {}
            if spec.version == "v9_1"
            else {"failed_expansion_reward": 0.0}
        ),
        "root_expansion": spec.version == "v9_1",
        "ancestor_history_limit": 8,
        "direct_child_limit": 8,
        "direct_child_top_count": 4,
        "reference_temperature": 0.2,
        "exploration_constant": 0.1,
        **(
            {"alpha": 0.5}
            if spec.version == "v9_1"
            else {"expansion_prior_weight": 1.0}
        ),
        "maximize": True,
        "operators": operator_names,
        "max_consecutive_sample_failures": 20,
        "max_stalled_iterations": 20,
        "checkpoint_interval": 10,
        "code_max_tokens": spec.llm_output_tokens,
        "context_token_limit": spec.context_token_limit,
        "random_seed": spec.seed,
    }
    if spec.version in {"v9", "v9_1"}:
        expected_method_params["history_protocol"] = "matched_history"
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
    _, task_kwargs = build_task(spec.task, spec.eval_workers)
    if spec.version == "v4":
        weights = V4ValueWeights()
    elif spec.version == "v5":
        weights = V5ValueWeights()
    else:
        weights = None
    method_params: dict[str, object]
    if spec.version in {"v8", "v9", "v9_1"}:
        method_params = {
            "protocol_id": (
                V8_PROTOCOL_ID
                if spec.version == "v8"
                else V9_PROTOCOL_ID
                if spec.version == "v9"
                else V91_PROTOCOL_ID
            ),
            "checkpoint_schema_version": (
                V8_CHECKPOINT_VERSION
                if spec.version == "v8"
                else V9_CHECKPOINT_VERSION
                if spec.version == "v9"
                else V91_CHECKPOINT_VERSION
            ),
            "max_sample_nums": spec.budget,
            "n_init": spec.n_init,
            "offspring_per_iteration": 1 if spec.version == "v9_1" else 2,
            "generation_protocol": "direct_code",
            "quality_normalization": (
                "mcts_minmax_q"
                if spec.version == "v9_1"
                else "global_midrank_percentile"
            ),
            "expansion_policy": (
                "progressive_widening_uct"
                if spec.version == "v9_1"
                else "adaptive_new_child_uct"
            ),
            "expansion_reward": (
                "child_continuation_value"
                if spec.version == "v9_1"
                else "batch_subtree_best_midrank"
            ),
            **(
                {}
                if spec.version == "v9_1"
                else {"failed_expansion_reward": 0.0}
            ),
            "root_expansion": spec.version == "v9_1",
            "ancestor_history_limit": 8,
            "direct_child_limit": 8,
            "direct_child_top_count": 4,
            "reference_temperature": 0.2,
            "exploration_constant": 0.1,
            **(
                {"alpha": 0.5}
                if spec.version == "v9_1"
                else {"expansion_prior_weight": 1.0}
            ),
            "max_consecutive_sample_failures": 20,
            "max_stalled_iterations": 20,
            "checkpoint_interval": 10,
        }
        if spec.version in {"v9", "v9_1"}:
            method_params["history_protocol"] = "matched_history"
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
    if spec.version in {"v8", "v9", "v9_1"}:
        method_params.update(
            {
                "maximize": True,
                "operators": V8_OPERATOR_NAMES
                if spec.version == "v8"
                else V9_OPERATOR_NAMES
                if spec.version == "v9"
                else V91_OPERATOR_NAMES,
                "code_max_tokens": spec.llm_output_tokens,
                "context_token_limit": spec.context_token_limit,
                "random_seed": spec.seed,
            }
        )
    write_run_config_file(
        run_dir,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "run_dir": str(run_dir),
            "task": spec.task,
            "method": spec.method_name,
            "timestamp": run_name,
            "repeat": spec.repeat,
            "backend": spec.backend,
            "llm": llm_payload(
                base_url=spec.base_url,
                model=spec.model,
                no_proxy=spec.no_proxy,
                max_tokens=spec.llm_output_tokens,
                temperature=1.0,
            ),
            "task_eval": task_kwargs,
            "method_params": method_params,
        },
    )


def run_experiment(spec: RunSpec) -> Path:
    run_dir, run_name, resumed = resolve_run_dir(spec)
    if not resumed:
        write_run_config(spec, run_dir, run_name)

    print(f"run_dir={run_dir}")
    resume_source = checkpoint_source(spec, run_dir) if resumed else None
    run_in_tmux_log(
        run_dir,
        run_dir / "logs",
        [],
        lambda: build_method(spec, run_dir, resume_source).run(),
    )
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

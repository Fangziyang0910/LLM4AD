"""Run one TraceAAD experiment with an explicit task and version."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from llm4ad.method.traceaad_v9_7 import (
    INITIAL_ROOT_COUNT as V97_INITIAL_ROOT_COUNT,
    LOGICAL_MODEL_NAME as V97_LOGICAL_MODEL_NAME,
    MAX_HISTORY_EVENTS as V97_MAX_HISTORY_EVENTS,
    REFINE_PROBABILITY as V97_REFINE_PROBABILITY,
    RunArtifacts as V97RunArtifacts,
    TraceAADV97,
)
from llm4ad.method.traceaad_v9_14 import (
    INITIAL_ROOT_COUNT as V914_INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS as V914_MAX_HISTORY_EVENTS,
    REFINE_PROBABILITY as V914_REFINE_PROBABILITY,
    RunArtifacts as V914RunArtifacts,
    TraceAADV914,
)
from llm4ad.method.traceaad_v9_15 import (
    BASE_EXPLORE_PROBABILITY as V915_BASE_EXPLORE_PROBABILITY,
    BONUS_CAP_SCALE as V915_BONUS_CAP_SCALE,
    ESS_FRACTION as V915_ESS_FRACTION,
    EXPLORE_PROBABILITY_MAX as V915_EXPLORE_PROBABILITY_MAX,
    EXPLORE_PROBABILITY_MIN as V915_EXPLORE_PROBABILITY_MIN,
    INITIAL_ROOT_COUNT as V915_INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS as V915_MAX_HISTORY_EVENTS,
    MIN_ESS_TARGET as V915_MIN_ESS_TARGET,
    STAGNATION_GAIN as V915_STAGNATION_GAIN,
    STAGNATION_WINDOW as V915_STAGNATION_WINDOW,
    TRAJECTORY_WINDOW as V915_TRAJECTORY_WINDOW,
    RunArtifacts as V915RunArtifacts,
    TraceAADV915,
)
from llm4ad.method.traceaad_v9_16 import (
    ESS_FRACTION as V916_ESS_FRACTION,
    EXPLORE_PROBABILITY as V916_EXPLORE_PROBABILITY,
    INITIAL_ROOT_COUNT as V916_INITIAL_ROOT_COUNT,
    LANDING_HORIZON as V916_LANDING_HORIZON,
    LANDING_PROBABILITY as V916_LANDING_PROBABILITY,
    LANDING_RATIO as V916_LANDING_RATIO,
    MAX_HISTORY_EVENTS as V916_MAX_HISTORY_EVENTS,
    MIN_ESS_TARGET as V916_MIN_ESS_TARGET,
    REFINE_PROBABILITY as V916_REFINE_PROBABILITY,
    RunArtifacts as V916RunArtifacts,
    TraceAADV916,
)
from llm4ad.method.traceaad_v9_17 import (
    ACTIVE_CAPACITY as V917_ACTIVE_CAPACITY,
    BLOCK_HORIZON as V917_BLOCK_HORIZON,
    INITIAL_ROOT_COUNT as V917_INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS as V917_MAX_HISTORY_EVENTS,
    RunArtifacts as V917RunArtifacts,
    TraceAADV917,
)

from .._common import (
    ALL_TASKS,
    BACKENDS,
    EXPERIMENTS_ROOT,
    TASKS as TASKS,
    TaskName,
    build_llm_client,
    build_task,
    resolve_backend,
    resolve_run_dir as resolve_run_dir_file,
    run_in_tmux_log,
    write_run_config as write_run_config_file,
)

VersionName = Literal[
    "v9_7",
    "v9_14",
    "v9_15",
    "v9_16",
    "v9_17",
    "v9_17_fixed_cycle",
]

VERSIONS: tuple[VersionName, ...] = (
    "v9_7",
    "v9_14",
    "v9_15",
    "v9_16",
    "v9_17",
    "v9_17_fixed_cycle",
)
TRACEAAD_V915_VERSIONS = {"v9_15"}
TRACEAAD_V916_VERSIONS = {"v9_16"}
TRACEAAD_V917_VERSIONS = {"v9_17", "v9_17_fixed_cycle"}


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
    context_token_limit: int = 24576
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    resume_from: Path | None = None
    initialization_checkpoint: Path | None = None
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
        return 8192


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
    context_token_limit: int | None = None,
    seed: int = 0,
    repeat: int | None = None,
    run_name: str | None = None,
    resume_from: Path | None = None,
    initialization_checkpoint: Path | None = None,
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
        n_init=(
            V97_INITIAL_ROOT_COUNT
            if version == "v9_7"
            else V914_INITIAL_ROOT_COUNT
            if version == "v9_14"
            else V915_INITIAL_ROOT_COUNT
            if version in TRACEAAD_V915_VERSIONS
            else V916_INITIAL_ROOT_COUNT
            if version in TRACEAAD_V916_VERSIONS
            else V917_INITIAL_ROOT_COUNT
        )
        if n_init is None
        else n_init,
        eval_workers=eval_workers,
        output_tokens=output_tokens,
        context_token_limit=(
            32768
            if context_token_limit is None
            and version
            in {
                "v9_7",
                "v9_14",
                "v9_15",
                "v9_16",
                "v9_17",
                "v9_17_fixed_cycle",
            }
            else 24576
            if context_token_limit is None
            else context_token_limit
        ),
        seed=seed,
        repeat=repeat,
        run_name=run_name,
        resume_from=None if resume_from is None else resume_from.resolve(),
        initialization_checkpoint=(
            None
            if initialization_checkpoint is None
            else initialization_checkpoint.resolve()
        ),
        experiments_root=experiments_root.resolve(),
    )
    if spec.budget <= 0:
        raise ValueError("budget must be positive")
    if spec.n_init <= 0:
        raise ValueError("n_init must be positive")
    if spec.version == "v9_7" and spec.n_init != V97_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.7 requires exactly eight initial roots")
    if spec.version == "v9_14" and spec.n_init != V914_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.14 requires exactly eight initial roots")
    if spec.version in TRACEAAD_V915_VERSIONS and spec.n_init != V915_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.15 requires exactly eight initial roots")
    if spec.version in TRACEAAD_V916_VERSIONS and spec.n_init != V916_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.16 requires exactly eight initial roots")
    if spec.version in TRACEAAD_V917_VERSIONS and spec.n_init != V917_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.17 requires exactly eight initial roots")
    if spec.eval_workers is not None and spec.eval_workers <= 0:
        raise ValueError("eval_workers must be positive")
    if spec.llm_output_tokens <= 0:
        raise ValueError("output_tokens must be positive")
    if spec.context_token_limit <= 0:
        raise ValueError("context_token_limit must be positive")
    if spec.resume_from is not None and spec.run_name is not None:
        raise ValueError("run_name cannot be combined with resume_from")
    if spec.resume_from is not None and spec.initialization_checkpoint is not None:
        raise ValueError("resume_from cannot be combined with initialization_checkpoint")
    if (
        spec.initialization_checkpoint is not None
        and spec.version != "v9_17_fixed_cycle"
    ):
        raise ValueError(
            "initialization_checkpoint is only valid for V9.17 FixedCycle"
        )
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
    if spec.version == "v9_7":
        return TraceAADV97(
            llm=llm,
            evaluation=evaluation,
            artifacts=V97RunArtifacts(run_dir=run_dir),
            budget=spec.budget,
            n_roots=spec.n_init,
            max_tokens=spec.llm_output_tokens,
            max_history=V97_MAX_HISTORY_EVENTS,
            seed=spec.seed,
            resume_from=resume_from,
            checkpoint_dir=run_dir / "checkpoints",
        )
    if spec.version == "v9_14":
        return TraceAADV914(
            llm=llm,
            evaluation=evaluation,
            artifacts=V914RunArtifacts(run_dir=run_dir),
            budget=spec.budget,
            n_roots=spec.n_init,
            max_tokens=spec.llm_output_tokens,
            max_history=V914_MAX_HISTORY_EVENTS,
            seed=spec.seed,
            resume_from=resume_from,
            checkpoint_dir=run_dir / "checkpoints",
        )
    if spec.version == "v9_15":
        return TraceAADV915(
            llm=llm,
            evaluation=evaluation,
            artifacts=V915RunArtifacts(run_dir=run_dir),
            budget=spec.budget,
            n_roots=spec.n_init,
            max_tokens=spec.llm_output_tokens,
            max_history=V915_MAX_HISTORY_EVENTS,
            seed=spec.seed,
            resume_from=resume_from,
            checkpoint_dir=run_dir / "checkpoints",
        )
    if spec.version == "v9_16":
        return TraceAADV916(
            llm=llm,
            evaluation=evaluation,
            artifacts=V916RunArtifacts(run_dir=run_dir),
            budget=spec.budget,
            n_roots=spec.n_init,
            max_tokens=spec.llm_output_tokens,
            max_history=V916_MAX_HISTORY_EVENTS,
            seed=spec.seed,
            resume_from=resume_from,
            checkpoint_dir=run_dir / "checkpoints",
        )
    if spec.version in TRACEAAD_V917_VERSIONS:
        method_resume = resume_from
        fork_from_initialization = False
        if method_resume is None and spec.initialization_checkpoint is not None:
            _seed_paired_artifacts(spec.initialization_checkpoint, run_dir)
            method_resume = spec.initialization_checkpoint
            fork_from_initialization = True
        return TraceAADV917(
            llm=llm,
            evaluation=evaluation,
            artifacts=V917RunArtifacts(run_dir=run_dir),
            budget=spec.budget,
            n_roots=spec.n_init,
            max_tokens=spec.llm_output_tokens,
            max_history=V917_MAX_HISTORY_EVENTS,
            seed=spec.seed,
            resume_from=method_resume,
            checkpoint_dir=run_dir / "checkpoints",
            adaptive_sweeps=spec.version == "v9_17",
            fork_from_initialization=fork_from_initialization,
        )
    raise ValueError(f"unsupported TraceAAD version: {spec.version}")


def resolve_run_dir(spec: RunSpec) -> tuple[Path, str, bool]:
    if spec.resume_from is not None:
        run_dir = spec.resume_from
        if not run_dir.is_dir():
            raise FileNotFoundError(f"resume run directory does not exist: {run_dir}")
        _validate_resume_config(spec, run_dir)
        return run_dir, run_dir.name, True
    run_dir, run_name = resolve_run_dir_file(spec.experiment_root, spec.run_name)
    return run_dir, run_name, False


def _task_eval_protocol(task_eval: object) -> object:
    """Compare evaluation semantics, not local ACO worker count."""
    if not isinstance(task_eval, dict):
        return task_eval
    payload = dict(task_eval)
    payload.pop("n_workers", None)
    return json.loads(json.dumps(payload, sort_keys=True))


def _validate_resume_config(spec: RunSpec, run_dir: Path) -> None:
    path = run_dir / "run_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"resume config does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {"task": spec.task, "method": spec.method_name}
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"resume config mismatch: expected {expected}, found {actual}")
    _, task_kwargs = build_task(spec.task, spec.eval_workers)
    normalized_task_kwargs = json.loads(json.dumps(task_kwargs, sort_keys=True))
    if spec.version == "v9_7":
        expected_method_params = _v97_method_params(spec)
    elif spec.version == "v9_14":
        expected_method_params = _v914_method_params(spec)
    elif spec.version == "v9_16":
        expected_method_params = _v916_method_params(spec)
    elif spec.version in TRACEAAD_V917_VERSIONS:
        expected_method_params = _v917_method_params(spec)
    else:
        expected_method_params = _v915_method_params(spec)
    expected_protocol = {
        "task_eval": _task_eval_protocol(normalized_task_kwargs),
        "method_params": expected_method_params,
        "generator_environment": _versioned_generator_environment(spec),
    }
    actual_protocol = {
        "task_eval": _task_eval_protocol(payload.get("task_eval")),
        "method_params": {
            key: payload.get("method_params", {}).get(key)
            for key in expected_method_params
        },
        "generator_environment": payload.get("generator_environment"),
    }
    if actual_protocol != expected_protocol:
        raise ValueError(
            f"resume config mismatch for TraceAAD {spec.version.upper()}; "
            "use the original "
            "model, evaluation, budget, and context settings"
        )


def checkpoint_source(spec: RunSpec, run_dir: Path) -> Path:
    """Return the checkpoint path used for resume."""
    return run_dir / "checkpoints" / "latest.json"


def write_run_config(spec: RunSpec, run_dir: Path, run_name: str) -> None:
    _, task_kwargs = build_task(spec.task, spec.eval_workers)
    if spec.version == "v9_7":
        method_params = _v97_method_params(spec)
    elif spec.version == "v9_14":
        method_params = _v914_method_params(spec)
    elif spec.version == "v9_15":
        method_params = _v915_method_params(spec)
    elif spec.version == "v9_16":
        method_params = _v916_method_params(spec)
    else:
        method_params = _v917_method_params(spec)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "task": spec.task,
        "method": spec.method_name,
        "timestamp": run_name,
        "repeat": spec.repeat,
        "task_eval": task_kwargs,
        "method_params": method_params,
    }
    payload["generator_environment"] = _versioned_generator_environment(spec)
    if spec.initialization_checkpoint is not None:
        metadata_path = spec.initialization_checkpoint.parent / "complete.json"
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )
        payload["paired_initialization"] = {
            "checkpoint": str(spec.initialization_checkpoint),
            **metadata,
        }
    write_run_config_file(run_dir, payload)


def _versioned_logical_model_name(spec: RunSpec) -> str:
    model = spec.model.lower()
    if "qwen3.8" in model:
        return "Qwen3.8-27B"
    if spec.version == "v9_14":
        return "Qwen3.6-27B"
    if spec.version in (
        TRACEAAD_V915_VERSIONS
        | TRACEAAD_V916_VERSIONS
        | TRACEAAD_V917_VERSIONS
    ):
        return "Qwen3.6-27B"
    return V97_LOGICAL_MODEL_NAME


def _versioned_generator_environment(spec: RunSpec) -> dict[str, object]:
    return {
        "logical_model_name": _versioned_logical_model_name(spec),
        "temperature": 1.0,
        "max_new_tokens": spec.llm_output_tokens,
        "sampling_seed": spec.seed,
        "max_total_context": spec.context_token_limit,
    }


def _v97_method_params(spec: RunSpec) -> dict[str, object]:
    return {
        "budget": spec.budget,
        "n_roots": spec.n_init,
        "max_history": V97_MAX_HISTORY_EVENTS,
        "maximize": True,
        "max_tokens": spec.llm_output_tokens,
        "seed": spec.seed,
        "refine_probability": V97_REFINE_PROBABILITY,
        "explore_probability": 1.0 - V97_REFINE_PROBABILITY,
    }


def _v914_method_params(spec: RunSpec) -> dict[str, object]:
    return {
        "budget": spec.budget,
        "n_roots": spec.n_init,
        "max_history": V914_MAX_HISTORY_EVENTS,
        "maximize": True,
        "max_tokens": spec.llm_output_tokens,
        "seed": spec.seed,
        "refine_probability": V914_REFINE_PROBABILITY,
        "explore_probability": 1.0 - V914_REFINE_PROBABILITY,
    }


def _v915_method_params(spec: RunSpec) -> dict[str, object]:
    return {
        "budget": spec.budget,
        "n_roots": spec.n_init,
        "max_history": V915_MAX_HISTORY_EVENTS,
        "maximize": True,
        "max_tokens": spec.llm_output_tokens,
        "seed": spec.seed,
        "base_explore_probability": V915_BASE_EXPLORE_PROBABILITY,
        "stagnation_window": V915_STAGNATION_WINDOW,
        "stagnation_gain": V915_STAGNATION_GAIN,
        "explore_probability_min": V915_EXPLORE_PROBABILITY_MIN,
        "explore_probability_max": V915_EXPLORE_PROBABILITY_MAX,
        "bonus_cap_scale": V915_BONUS_CAP_SCALE,
        "trajectory_window": V915_TRAJECTORY_WINDOW,
        "ess_fraction": V915_ESS_FRACTION,
        "min_ess_target": V915_MIN_ESS_TARGET,
        "error_handling": True,
        "error_retries": 2,
        "retry_policy": "two_bounded_repairs",
        "retry_budget": "initial_candidates",
    }


def _v916_method_params(spec: RunSpec) -> dict[str, object]:
    return {
        "budget": spec.budget,
        "n_roots": spec.n_init,
        "max_history": V916_MAX_HISTORY_EVENTS,
        "maximize": True,
        "max_tokens": spec.llm_output_tokens,
        "seed": spec.seed,
        "refine_probability": V916_REFINE_PROBABILITY,
        "explore_probability": V916_EXPLORE_PROBABILITY,
        "ess_fraction": V916_ESS_FRACTION,
        "min_ess_target": V916_MIN_ESS_TARGET,
        "landing_ratio": V916_LANDING_RATIO,
        "landing_probability": V916_LANDING_PROBABILITY,
        "landing_horizon": V916_LANDING_HORIZON,
        "parent_score": "quality_only",
        "error_handling": True,
        "error_retries": 2,
        "retry_policy": "two_bounded_repairs",
        "retry_budget": "initial_candidates",
    }


def _v917_method_params(spec: RunSpec) -> dict[str, object]:
    params: dict[str, object] = {
        "budget": spec.budget,
        "n_roots": spec.n_init,
        "active_capacity": V917_ACTIVE_CAPACITY,
        "block_horizon": V917_BLOCK_HORIZON,
        "max_history": V917_MAX_HISTORY_EVENTS,
        "maximize": True,
        "max_tokens": spec.llm_output_tokens,
        "seed": spec.seed,
        "hypothesis_birth": "valid_root_or_explore",
        "competition_rank": "frontier_quality_then_creation",
        "development_continuation": "positive_block_gain",
        "refine_parent_score": "q_plus_frozen_scale_over_sqrt_count",
        "discovery_source": "highest_active_frontier",
        "error_handling": True,
        "error_retries": 2,
        "retry_policy": "two_bounded_repairs",
        "retry_budget": "primary_candidates",
    }
    if spec.version == "v9_17_fixed_cycle":
        params["development_continuation"] = "fixed_cycle_after_full_sweep"
    return params


def _seed_paired_artifacts(initialization_checkpoint: Path, run_dir: Path) -> None:
    if not initialization_checkpoint.is_file():
        raise FileNotFoundError(
            f"paired initialization checkpoint does not exist: {initialization_checkpoint}"
        )
    bundle = initialization_checkpoint.parent
    required = ("evaluations.csv", "mechanism_events.jsonl")
    optional = ("best_program.py",)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in (*required, *optional):
        source = bundle / name
        if not source.is_file():
            if name in required:
                raise FileNotFoundError(f"paired initialization artifact is missing: {source}")
            continue
        target = run_dir / name
        if target.is_file():
            if target.read_bytes() != source.read_bytes():
                raise ValueError(f"paired artifact already differs: {target}")
            continue
        shutil.copyfile(source, target)


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
    parser.add_argument("--task", required=True, choices=ALL_TASKS)
    parser.add_argument("--version", required=True, choices=VERSIONS)
    parser.add_argument("--backend", choices=tuple(BACKENDS), default="local")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--no-proxy")
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--n-init", type=int)
    parser.add_argument("--eval-workers", type=int)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--context-token-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--initialization-checkpoint", type=Path)
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
        context_token_limit=args.context_token_limit,
        seed=args.seed,
        repeat=args.repeat,
        run_name=args.run_name,
        resume_from=args.resume_from,
        initialization_checkpoint=args.initialization_checkpoint,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_experiment(spec_from_args(args))


if __name__ == "__main__":
    main()

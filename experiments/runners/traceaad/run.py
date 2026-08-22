"""Run one TraceAAD experiment with an explicit task and version."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from llm4ad.method.traceaad_v4 import (
    RunArtifacts as V4RunArtifacts,
    TraceAADV4,
    ValueWeights as V4ValueWeights,
)
from llm4ad.method.traceaad_v5 import (
    RunArtifacts as V5RunArtifacts,
    TraceAADV5,
    ValueWeights as V5ValueWeights,
)
from llm4ad.method.traceaad_v8 import (
    RunArtifacts as V8RunArtifacts,
    TraceAADV8,
)
from llm4ad.method.traceaad_v8.operators import DEFAULT_OPERATORS as V8_OPERATORS
from llm4ad.method.traceaad_v9 import (
    RunArtifacts as V9RunArtifacts,
    TraceAADV9,
)
from llm4ad.method.traceaad_v9.operators import DEFAULT_OPERATORS as V9_OPERATORS
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

from .._common import (
    ALL_TASKS,
    BACKENDS,
    EXPERIMENTS_ROOT,
    TASKS as TASKS,
    TaskName,
    build_llm_client,
    build_task,
    llm_payload,
    resolve_backend,
    resolve_run_dir as resolve_run_dir_file,
    run_in_tmux_log,
    write_run_config as write_run_config_file,
)

VersionName = Literal[
    "v4",
    "v5",
    "v8",
    "v9",
    "v9_7",
    "v9_14",
    "v9_15",
    "v9_16",
]

VERSIONS: tuple[VersionName, ...] = (
    "v4",
    "v5",
    "v8",
    "v9",
    "v9_7",
    "v9_14",
    "v9_15",
    "v9_16",
)
TRACEAAD_V915_VERSIONS = {"v9_15"}
TRACEAAD_V916_VERSIONS = {"v9_16"}
V8_OPERATOR_NAMES = [str(operator_type.name) for operator_type in V8_OPERATORS]
V9_OPERATOR_NAMES = [str(operator_type.name) for operator_type in V9_OPERATORS]


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
    context_token_limit: int | None = None,
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
        n_init=(
            V97_INITIAL_ROOT_COUNT
            if version == "v9_7"
            else V914_INITIAL_ROOT_COUNT
            if version == "v9_14"
            else V915_INITIAL_ROOT_COUNT
            if version in TRACEAAD_V915_VERSIONS
            else V916_INITIAL_ROOT_COUNT
            if version in TRACEAAD_V916_VERSIONS
            else 10
            if version in {"v8", "v9"}
            else 30
        )
        if n_init is None
        else n_init,
        eval_workers=eval_workers,
        output_tokens=output_tokens,
        action_max_tokens=action_max_tokens,
        context_token_limit=(
            32768
            if context_token_limit is None
            and version
            in {
                "v9_7",
                "v9_14",
                "v9_15",
                "v9_16",
            }
            else 24576
            if context_token_limit is None
            else context_token_limit
        ),
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
    if spec.version == "v9_7" and spec.n_init != V97_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.7 requires exactly eight initial roots")
    if spec.version == "v9_14" and spec.n_init != V914_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.14 requires exactly eight initial roots")
    if spec.version in TRACEAAD_V915_VERSIONS and spec.n_init != V915_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.15 requires exactly eight initial roots")
    if spec.version in TRACEAAD_V916_VERSIONS and spec.n_init != V916_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.16 requires exactly eight initial roots")
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
    common = {
        "llm": llm,
        "evaluation": evaluation,
        "max_sample_nums": spec.budget,
        "n_init": spec.n_init,
        "max_stalled_iterations": 20,
        "checkpoint_interval": 10,
        "resume_from": resume_from,
        "checkpoint_dir": run_dir / "checkpoints",
    }
    if spec.version in {"v8", "v9"}:
        method_type = TraceAADV8 if spec.version == "v8" else TraceAADV9
        artifacts_type = V8RunArtifacts if spec.version == "v8" else V9RunArtifacts
        return method_type(
            artifacts=artifacts_type(run_dir=run_dir),
            ancestor_history_limit=8,
            direct_child_limit=8,
            direct_child_top_count=4,
            code_max_tokens=spec.llm_output_tokens,
            context_token_limit=spec.context_token_limit,
            random_seed=spec.seed,
            offspring_per_iteration=2,
            reference_temperature=0.2,
            exploration_constant=0.1,
            expansion_prior_weight=1.0,
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
            artifacts=V4RunArtifacts(run_dir=run_dir),
            value_weights=V4ValueWeights(),
            **population_common,
            **common,
        )
    return TraceAADV5(
        artifacts=V5RunArtifacts(run_dir=run_dir),
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
    if spec.version not in {
        "v8",
        "v9",
        "v9_7",
        "v9_14",
        "v9_15",
        "v9_16",
    }:
        return
    _, task_kwargs = build_task(spec.task, spec.eval_workers)
    normalized_task_kwargs = json.loads(json.dumps(task_kwargs, sort_keys=True))
    if spec.version in {
        "v9_7",
        "v9_14",
        "v9_15",
        "v9_16",
    }:
        if spec.version == "v9_7":
            expected_method_params = _v97_method_params(spec)
        elif spec.version == "v9_14":
            expected_method_params = _v914_method_params(spec)
        elif spec.version == "v9_16":
            expected_method_params = _v916_method_params(spec)
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
        return
    if spec.version == "v8":
        operator_names = V8_OPERATOR_NAMES
    else:
        operator_names = V9_OPERATOR_NAMES
    expected_method_params = {
        "max_sample_nums": spec.budget,
        "n_init": spec.n_init,
        "generation_protocol": "direct_code",
        "offspring_per_iteration": 2,
        "quality_normalization": "global_midrank_percentile",
        "expansion_policy": "adaptive_new_child_uct",
        "expansion_reward": "batch_subtree_best_midrank",
        "failed_expansion_reward": 0.0,
        "root_expansion": False,
        "reference_temperature": 0.2,
        "exploration_constant": 0.1,
        "expansion_prior_weight": 1.0,
        "ancestor_history_limit": 8,
        "direct_child_limit": 8,
        "direct_child_top_count": 4,
        "maximize": True,
        "operators": operator_names,
        "max_stalled_iterations": 20,
        "checkpoint_interval": 10,
        "code_max_tokens": spec.llm_output_tokens,
        "context_token_limit": spec.context_token_limit,
        "random_seed": spec.seed,
    }
    if spec.version == "v9":
        expected_method_params["history_protocol"] = "matched_history"
    expected_protocol = {
        "backend": spec.backend,
        "task_eval": _task_eval_protocol(normalized_task_kwargs),
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
        "task_eval": _task_eval_protocol(payload.get("task_eval")),
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
    if spec.version == "v9_7":
        method_params = _v97_method_params(spec)
    elif spec.version == "v9_14":
        method_params = _v914_method_params(spec)
    elif spec.version == "v9_15":
        method_params = _v915_method_params(spec)
    elif spec.version == "v9_16":
        method_params = _v916_method_params(spec)
    elif spec.version in {"v8", "v9"}:
        method_params = {
            "max_sample_nums": spec.budget,
            "n_init": spec.n_init,
            "generation_protocol": "direct_code",
            "offspring_per_iteration": 2,
            "quality_normalization": "global_midrank_percentile",
            "expansion_policy": "adaptive_new_child_uct",
            "expansion_reward": "batch_subtree_best_midrank",
            "failed_expansion_reward": 0.0,
            "root_expansion": False,
            "reference_temperature": 0.2,
            "exploration_constant": 0.1,
            "expansion_prior_weight": 1.0,
            "ancestor_history_limit": 8,
            "direct_child_limit": 8,
            "direct_child_top_count": 4,
            "max_stalled_iterations": 20,
            "checkpoint_interval": 10,
        }
        if spec.version == "v9":
            method_params["history_protocol"] = "matched_history"
    else:
        method_params = {
            "max_sample_nums": spec.budget,
            "n_init": spec.n_init,
            "actions_per_iteration": 2,
            "max_trajectory_length": 8,
            "max_active_trajectories": 30,
            "softmax_temperature": 0.2,
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
    if spec.version in {"v8", "v9"}:
        method_params.update(
            {
                "maximize": True,
                "operators": (
                    V8_OPERATOR_NAMES if spec.version == "v8" else V9_OPERATOR_NAMES
                ),
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
        "timestamp": run_name,
        "repeat": spec.repeat,
        "task_eval": task_kwargs,
        "method_params": method_params,
    }
    if spec.version in {
        "v9_7",
        "v9_14",
        "v9_15",
        "v9_16",
    }:
        payload["generator_environment"] = _versioned_generator_environment(spec)
    else:
        payload["backend"] = spec.backend
        payload["llm"] = llm_payload(
            base_url=spec.base_url,
            model=spec.model,
            no_proxy=spec.no_proxy,
            max_tokens=spec.llm_output_tokens,
            temperature=1.0,
        )
    write_run_config_file(run_dir, payload)


def _versioned_logical_model_name(spec: RunSpec) -> str:
    model = spec.model.lower()
    if "qwen3.8" in model:
        return "Qwen3.8-27B"
    if spec.version == "v9_14":
        return "Qwen3.6-27B"
    if spec.version in TRACEAAD_V915_VERSIONS | TRACEAAD_V916_VERSIONS:
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
    parser.add_argument("--action-max-tokens", type=int, default=1024)
    parser.add_argument("--context-token-limit", type=int)
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

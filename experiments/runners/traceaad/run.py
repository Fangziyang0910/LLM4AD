"""Run one TraceAAD experiment with an explicit task and version."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
    CHECKPOINT_VERSION as V8_CHECKPOINT_VERSION,
    PROTOCOL_ID as V8_PROTOCOL_ID,
    RunArtifacts as V8RunArtifacts,
    TraceAADV8,
)
from llm4ad.method.traceaad_v8.operators import DEFAULT_OPERATORS as V8_OPERATORS
from llm4ad.method.traceaad_v9 import (
    CHECKPOINT_VERSION as V9_CHECKPOINT_VERSION,
    PROTOCOL_ID as V9_PROTOCOL_ID,
    RunArtifacts as V9RunArtifacts,
    TraceAADV9,
)
from llm4ad.method.traceaad_v9.operators import DEFAULT_OPERATORS as V9_OPERATORS
from llm4ad.method.traceaad_v9_7 import (
    CHECKPOINT_VERSION as V97_CHECKPOINT_VERSION,
    INITIAL_ROOT_COUNT as V97_INITIAL_ROOT_COUNT,
    LOGICAL_MODEL_NAME as V97_LOGICAL_MODEL_NAME,
    MAX_HISTORY_EVENTS as V97_MAX_HISTORY_EVENTS,
    PROTOCOL_ID as V97_PROTOCOL_ID,
    REFINE_PROBABILITY as V97_REFINE_PROBABILITY,
    RunArtifacts as V97RunArtifacts,
    TraceAADV97,
)
from llm4ad.method.traceaad_v9_8 import (
    CHECKPOINT_VERSION as V98_CHECKPOINT_VERSION,
    DEFAULT_MAX_CONSECUTIVE_ERRORS as V98_DEFAULT_MAX_CONSECUTIVE_ERRORS,
    DEFAULT_MAX_RESPONSES as V98_DEFAULT_MAX_RESPONSES,
    INITIAL_ROOT_COUNT as V98_INITIAL_ROOT_COUNT,
    LOGICAL_MODEL_NAME as V98_LOGICAL_MODEL_NAME,
    MAX_HISTORY_EVENTS as V98_MAX_HISTORY_EVENTS,
    PROTOCOL_ID as V98_PROTOCOL_ID,
    REFINE_PROBABILITY as V98_REFINE_PROBABILITY,
    SCORE_FORMULA_VERSION as V98_SCORE_FORMULA_VERSION,
    AllocationPolicy as V98AllocationPolicy,
    RunArtifacts as V98RunArtifacts,
    TraceAADV98,
)

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

VersionName = Literal["v4", "v5", "v8", "v9", "v9_7", "v9_8"]

VERSIONS: tuple[VersionName, ...] = (
    "v4",
    "v5",
    "v8",
    "v9",
    "v9_7",
    "v9_8",
)
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
    allocation_policy: str = V98AllocationPolicy.FULL.value
    max_responses: int = V98_DEFAULT_MAX_RESPONSES
    max_consecutive_errors: int = V98_DEFAULT_MAX_CONSECUTIVE_ERRORS
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
    allocation_policy: str = V98AllocationPolicy.FULL.value,
    max_responses: int = V98_DEFAULT_MAX_RESPONSES,
    max_consecutive_errors: int = V98_DEFAULT_MAX_CONSECUTIVE_ERRORS,
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
            else V98_INITIAL_ROOT_COUNT
            if version == "v9_8"
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
            if context_token_limit is None and version in {"v9_7", "v9_8"}
            else 24576
            if context_token_limit is None
            else context_token_limit
        ),
        seed=seed,
        repeat=repeat,
        run_name=run_name,
        resume_from=None if resume_from is None else resume_from.resolve(),
        allocation_policy=allocation_policy,
        max_responses=max_responses,
        max_consecutive_errors=max_consecutive_errors,
        experiments_root=experiments_root.resolve(),
    )
    if spec.budget <= 0:
        raise ValueError("budget must be positive")
    if spec.n_init <= 0:
        raise ValueError("n_init must be positive")
    if spec.version == "v9_7" and spec.n_init != V97_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.7 requires exactly eight initial roots")
    if spec.version == "v9_8" and spec.n_init != V98_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.8 requires exactly eight initial roots")
    if spec.version == "v9_8":
        V98AllocationPolicy(spec.allocation_policy)
        if spec.max_responses <= 0 or spec.max_consecutive_errors <= 0:
            raise ValueError("V9.8 safety limits must be positive")
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
            context_limit=spec.context_token_limit,
            max_history=V97_MAX_HISTORY_EVENTS,
            seed=spec.seed,
            resume_from=resume_from,
            checkpoint_dir=run_dir / "checkpoints",
        )
    if spec.version == "v9_8":
        return TraceAADV98(
            llm=llm,
            evaluation=evaluation,
            artifacts=V98RunArtifacts(run_dir=run_dir),
            budget=spec.budget,
            n_roots=spec.n_init,
            max_tokens=spec.llm_output_tokens,
            context_limit=spec.context_token_limit,
            max_history=V98_MAX_HISTORY_EVENTS,
            seed=spec.seed,
            allocation_policy=spec.allocation_policy,
            max_responses=spec.max_responses,
            max_consecutive_errors=spec.max_consecutive_errors,
            resume_from=resume_from,
            checkpoint_dir=run_dir / "checkpoints",
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
        "v8", "v9", "v9_7", "v9_8"
    }:
        return
    _, task_kwargs = build_task(spec.task, spec.eval_workers)
    normalized_task_kwargs = json.loads(json.dumps(task_kwargs, sort_keys=True))
    if spec.version in {"v9_7", "v9_8"}:
        expected_method_params = (
            _v97_method_params(spec) if spec.version == "v9_7" else _v98_method_params(spec)
        )
        expected_protocol = {
            "task_eval": normalized_task_kwargs,
            "method_params": expected_method_params,
            "generator_environment": _versioned_generator_environment(spec),
        }
        actual_protocol = {
            "task_eval": payload.get("task_eval"),
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
        protocol_id = V8_PROTOCOL_ID
        checkpoint_version = V8_CHECKPOINT_VERSION
        operator_names = V8_OPERATOR_NAMES
    else:
        protocol_id = V9_PROTOCOL_ID
        checkpoint_version = V9_CHECKPOINT_VERSION
        operator_names = V9_OPERATOR_NAMES
    expected_method_params = {
        "protocol_id": protocol_id,
        "checkpoint_schema_version": checkpoint_version,
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
        "max_consecutive_sample_failures": 20,
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
    if spec.version == "v9_7":
        method_params = _v97_method_params(spec)
    elif spec.version == "v9_8":
        method_params = _v98_method_params(spec)
    elif spec.version in {"v8", "v9"}:
        method_params = {
            "protocol_id": (
                V8_PROTOCOL_ID if spec.version == "v8" else V9_PROTOCOL_ID
            ),
            "checkpoint_schema_version": (
                V8_CHECKPOINT_VERSION
                if spec.version == "v8"
                else V9_CHECKPOINT_VERSION
            ),
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
            "max_consecutive_sample_failures": 20,
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
    if spec.version in {"v9_7", "v9_8"}:
        payload["generator_environment"] = _versioned_generator_environment(spec)
        if spec.version == "v9_8":
            payload["implementation"] = _v98_implementation_identity()
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
    return V98_LOGICAL_MODEL_NAME if spec.version == "v9_8" else V97_LOGICAL_MODEL_NAME


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
        "protocol_id": V97_PROTOCOL_ID,
        "checkpoint_schema_version": V97_CHECKPOINT_VERSION,
        "budget": spec.budget,
        "n_roots": spec.n_init,
        "max_history": V97_MAX_HISTORY_EVENTS,
        "maximize": True,
        "max_tokens": spec.llm_output_tokens,
        "context_limit": spec.context_token_limit,
        "seed": spec.seed,
        "refine_probability": V97_REFINE_PROBABILITY,
        "explore_probability": 1.0 - V97_REFINE_PROBABILITY,
    }


def _v98_method_params(spec: RunSpec) -> dict[str, object]:
    return {
        "protocol_id": V98_PROTOCOL_ID,
        "checkpoint_schema_version": V98_CHECKPOINT_VERSION,
        "score_formula_version": V98_SCORE_FORMULA_VERSION,
        "allocation_policy": spec.allocation_policy,
        "budget": spec.budget,
        "n_roots": spec.n_init,
        "max_history": V98_MAX_HISTORY_EVENTS,
        "maximize": True,
        "max_tokens": spec.llm_output_tokens,
        "context_limit": spec.context_token_limit,
        "seed": spec.seed,
        "refine_probability": V98_REFINE_PROBABILITY,
        "explore_probability": 1.0 - V98_REFINE_PROBABILITY,
        "max_responses": spec.max_responses,
        "max_consecutive_errors": spec.max_consecutive_errors,
    }


def _v98_implementation_identity() -> dict[str, object]:
    source_root = Path(__file__).resolve().parents[3]
    paths = sorted((source_root / "llm4ad" / "method" / "traceaad_v9_8").glob("*.py"))
    paths.extend(
        [
            Path(__file__).resolve(),
            Path(__file__).with_name("v98_mechanism_probe.py"),
            Path(__file__).with_name("v98_continuation_probe.py"),
        ]
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(source_root)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {
        "git_commit": commit,
        "worktree_dirty": dirty,
        "protocol_source_sha256": digest.hexdigest(),
        "source_files": [str(path.relative_to(source_root)) for path in paths],
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
    parser.add_argument("--context-token-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--allocation-policy",
        choices=tuple(item.value for item in V98AllocationPolicy),
        default=V98AllocationPolicy.FULL.value,
    )
    parser.add_argument("--max-responses", type=int, default=V98_DEFAULT_MAX_RESPONSES)
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=V98_DEFAULT_MAX_CONSECUTIVE_ERRORS,
    )
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
        allocation_policy=args.allocation_policy,
        max_responses=args.max_responses,
        max_consecutive_errors=args.max_consecutive_errors,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_experiment(spec_from_args(args))


if __name__ == "__main__":
    main()

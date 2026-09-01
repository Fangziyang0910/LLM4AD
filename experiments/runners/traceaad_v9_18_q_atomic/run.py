"""Run one TraceAAD V9.18 q_atomic experiment with an explicit task."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from llm4ad.method.traceaad_v9_18 import (
    ESS_FRACTION as V918_ESS_FRACTION,
    GLOBAL_FACTS_WINDOW as V918_GLOBAL_FACTS_WINDOW,
    INITIAL_ROOT_COUNT as V918_INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS as V918_MAX_HISTORY_EVENTS,
    MIN_ESS_TARGET as V918_MIN_ESS_TARGET,
    OPPORTUNITY_LAMBDA as V918_OPPORTUNITY_LAMBDA,
    OPPORTUNITY_TAU as V918_OPPORTUNITY_TAU,
    REFINE_PROBABILITY as V918_REFINE_PROBABILITY,
    EXPLORE_PROBABILITY as V918_EXPLORE_PROBABILITY,
    RunArtifacts as V918RunArtifacts,
    TraceAADV918,
)

from .._common import (
    ALL_TASKS,
    BACKENDS,
    EXPERIMENTS_ROOT,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_K,
    SAMPLING_TOP_P,
    TaskName,
    build_llm_client,
    build_task,
    llm_payload,
    resolve_backend,
    resolve_run_dir as resolve_run_dir_file,
    run_in_tmux_log,
    write_run_config as write_run_config_file,
)


@dataclass(frozen=True, slots=True)
class RunSpec:
    task: TaskName
    backend: str
    base_url: str
    model: str
    no_proxy: str
    n_init: int
    budget: int = 1000
    eval_workers: int | None = None
    output_tokens: int | None = None
    context_token_limit: int = 32768
    seed: int = 0
    repeat: int | None = None
    run_name: str | None = None
    resume_from: Path | None = None
    initialization_checkpoint: Path | None = None
    experiments_root: Path = EXPERIMENTS_ROOT

    @property
    def method_name(self) -> str:
        return "traceaad_v9_18_q_atomic"

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
        backend=backend,
        base_url=profile.base_url,
        model=profile.model,
        no_proxy=profile.no_proxy,
        budget=budget,
        n_init=V918_INITIAL_ROOT_COUNT if n_init is None else n_init,
        eval_workers=eval_workers,
        output_tokens=output_tokens,
        context_token_limit=32768 if context_token_limit is None else context_token_limit,
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
    if spec.n_init != V918_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.18 requires exactly eight initial roots")
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
    return spec


def build_method(
    spec: RunSpec,
    run_dir: Path,
    resume_from: Path | None = None,
):
    evaluation, task_kwargs = build_task(spec.task, spec.eval_workers)
    llm = build_llm_client(
        base_url=spec.base_url,
        model=spec.model,
        no_proxy=spec.no_proxy,
        max_tokens=spec.llm_output_tokens,
        temperature=1.0,
    )
    method_resume = resume_from
    fork_from_initialization = False
    if method_resume is None and spec.initialization_checkpoint is not None:
        _seed_paired_artifacts(spec.initialization_checkpoint, run_dir)
        method_resume = spec.initialization_checkpoint
        fork_from_initialization = True
    return TraceAADV918(
        llm=llm,
        evaluation=evaluation,
        artifacts=V918RunArtifacts(run_dir=run_dir),
        budget=spec.budget,
        n_roots=spec.n_init,
        max_tokens=spec.llm_output_tokens,
        max_history=V918_MAX_HISTORY_EVENTS,
        seed=spec.seed,
        resume_from=method_resume,
        checkpoint_dir=run_dir / "checkpoints",
        allocation_mode="q",
        explore_context="legacy",
        fork_from_initialization=fork_from_initialization,
    )


def _seed_paired_artifacts(initialization_checkpoint: Path, run_dir: Path) -> None:
    if not initialization_checkpoint.is_file():
        raise FileNotFoundError(
            f"paired initialization checkpoint does not exist: {initialization_checkpoint}"
        )
    bundle = initialization_checkpoint.parent
    required = ("evaluations.csv", "mechanism_events.jsonl")
    optional = ("best_program.py",)
    if not all((bundle / name).is_file() for name in required):
        parent_bundle = bundle.parent
        if all((parent_bundle / name).is_file() for name in required):
            bundle = parent_bundle
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in (*required, *optional):
        source = bundle / name
        if not source.is_file():
            if name in required:
                raise FileNotFoundError(f"paired artifact is missing: {source}")
            continue
        target = run_dir / name
        if target.is_file():
            if target.read_bytes() != source.read_bytes():
                raise ValueError(f"paired artifact already differs: {target}")
            continue
        shutil.copyfile(source, target)


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
    expected_method_params = _method_params(spec)
    expected_protocol = {
        "task_eval": _task_eval_protocol(normalized_task_kwargs),
        "method_params": expected_method_params,
        "generator_environment": _generator_environment(spec),
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
            "resume config mismatch for TraceAAD V9_18_Q_ATOMIC; "
            "use the original model, evaluation, budget, and context settings"
        )


def checkpoint_source(spec: RunSpec, run_dir: Path) -> Path:
    """Return the checkpoint path used for resume."""
    return run_dir / "checkpoints" / "latest.json"


def write_run_config(spec: RunSpec, run_dir: Path, run_name: str) -> None:
    _, task_kwargs = build_task(spec.task, spec.eval_workers)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "task": spec.task,
        "method": spec.method_name,
        "timestamp": run_name,
        "repeat": spec.repeat,
        "task_eval": task_kwargs,
        "method_params": _method_params(spec),
    }
    # V9.18 artifacts need explicit service metadata for held-out provenance;
    # retain the established config shape of earlier TraceAAD versions.
    payload.update(
        {
            "run_name": run_name,
            "backend": spec.backend,
            "seed": spec.seed,
            "llm": llm_payload(
                base_url=spec.base_url,
                model=spec.model,
                no_proxy=spec.no_proxy,
                max_tokens=spec.llm_output_tokens,
                temperature=1.0,
            ),
        }
    )

    payload["generator_environment"] = _generator_environment(spec)
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


def _logical_model_name(spec: RunSpec) -> str:
    if "qwen3.8" in spec.model.lower():
        return "Qwen3.8-27B"
    return "Qwen3.6-27B"


def _generator_environment(spec: RunSpec) -> dict[str, object]:
    return {
        "logical_model_name": _logical_model_name(spec),
        "temperature": SAMPLING_TEMPERATURE,
        "top_p": SAMPLING_TOP_P,
        "top_k": SAMPLING_TOP_K,
        "max_new_tokens": spec.llm_output_tokens,
        "sampling_seed": spec.seed,
        "max_total_context": spec.context_token_limit,
    }


def _method_params(spec: RunSpec) -> dict[str, object]:
    return {
        "budget": spec.budget,
        "n_roots": spec.n_init,
        "max_history": V918_MAX_HISTORY_EVENTS,
        "maximize": True,
        "max_tokens": spec.llm_output_tokens,
        "seed": spec.seed,
        "refine_probability": V918_REFINE_PROBABILITY,
        "explore_probability": V918_EXPLORE_PROBABILITY,
        "allocation_mode": "q",
        "explore_context": "legacy",
        "opportunity_lambda": V918_OPPORTUNITY_LAMBDA,
        "opportunity_tau": V918_OPPORTUNITY_TAU,
        "global_facts_window": V918_GLOBAL_FACTS_WINDOW,
        "ess_fraction": V918_ESS_FRACTION,
        "min_ess_target": V918_MIN_ESS_TARGET,
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
        description="Run one TraceAAD V9.18 q_atomic experiment with an explicit task."
    )
    parser.add_argument("--task", required=True, choices=ALL_TASKS)
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
    parser.add_argument("--experiments-root", type=Path, default=EXPERIMENTS_ROOT)
    return parser


def spec_from_args(args: argparse.Namespace) -> RunSpec:
    return make_run_spec(
        task=args.task,
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
        experiments_root=args.experiments_root,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_experiment(spec_from_args(args))


if __name__ == "__main__":
    main()

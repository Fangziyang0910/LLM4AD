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
from llm4ad.method.traceaad_v9_19 import (
    BEHAVESIM_PROTOCOL_ID as V919_BEHAVESIM_PROTOCOL_ID,
    CROSSOVER_PROBABILITY as V919_CROSSOVER_PROBABILITY,
    EXPLORE_MAX as V919_EXPLORE_MAX,
    EXPLORE_MIN as V919_EXPLORE_MIN,
    EXPLORE_NEUTRAL as V919_EXPLORE_NEUTRAL,
    INITIAL_ROOT_COUNT as V919_INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS as V919_MAX_HISTORY_EVENTS,
    MAX_REPAIRS as V919_MAX_REPAIRS,
    TRACKED_EVALUATIONS as V919_TRACKED_EVALUATIONS,
    TRAJECTORY_WINDOW as V919_TRAJECTORY_WINDOW,
    RunArtifacts as V919RunArtifacts,
    TraceAADV919,
    W_PROMISE as V919_W_PROMISE,
    W_TRAJECTORY as V919_W_TRAJECTORY,
    W_UNDERDEVELOPMENT as V919_W_UNDERDEVELOPMENT,
)
from llm4ad.method.traceaad_v9_20 import (
    ACTION_TEMPERATURE as V920_ACTION_TEMPERATURE,
    BEHAVESIM_PROTOCOL_ID as V920_BEHAVESIM_PROTOCOL_ID,
    COVERAGE_MIX as V920_COVERAGE_MIX,
    ESS_FRACTION as V920_ESS_FRACTION,
    EXPLORE_MAX as V920_EXPLORE_MAX,
    EXPLORE_MIN as V920_EXPLORE_MIN,
    EXPLORE_NEUTRAL as V920_EXPLORE_NEUTRAL,
    INITIAL_ROOT_COUNT as V920_INITIAL_ROOT_COUNT,
    MAX_REPAIRS as V920_MAX_REPAIRS,
    MAX_HISTORY_EVENTS as V920_MAX_HISTORY_EVENTS,
    MIN_ESS_TARGET as V920_MIN_ESS_TARGET,
    TRACKED_EVALUATIONS as V920_TRACKED_EVALUATIONS,
    RunArtifacts as V920RunArtifacts,
    TraceAADV920,
)
from llm4ad.method.traceaad_v9_21 import (
    BATCH_SIZE as V921_BATCH_SIZE,
    INITIAL_ROOT_COUNT as V921_INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS as V921_MAX_HISTORY_EVENTS,
    MAX_REPAIRS as V921_MAX_REPAIRS,
    REALIZATIONS_PER_IDEA as V921_REALIZATIONS_PER_IDEA,
    TraceAADV921,
    RunArtifacts as V921RunArtifacts,
)
from llm4ad.method.traceaad_v9_22 import (
    BATCH_SIZE as V922_BATCH_SIZE,
    IDEAS_PER_BATCH as V922_IDEAS_PER_BATCH,
    INITIAL_ROOT_COUNT as V922_INITIAL_ROOT_COUNT,
    MAX_HISTORY_EVENTS as V922_MAX_HISTORY_EVENTS,
    MAX_REPAIRS as V922_MAX_REPAIRS,
    REALIZATIONS_PER_IDEA as V922_REALIZATIONS_PER_IDEA,
    TraceAADV922,
    RunArtifacts as V922RunArtifacts,
)
from llm4ad.method.traceaad_v10 import (
    COMPETITIVE_SET_SIZE as V10_COMPETITIVE_SET_SIZE,
    FORMATION_WINDOW as V10_FORMATION_WINDOW,
    G_HORIZONS as V10_G_HORIZONS,
    INITIAL_ROOT_COUNT as V10_INITIAL_ROOT_COUNT,
    MAX_REPAIRS as V10_MAX_REPAIRS,
    REFERENCE_COUNT as V10_REFERENCE_COUNT,
    RESTART_CARDS as V10_RESTART_CARDS,
    SCREEN_SIZE as V10_SCREEN_SIZE,
    TraceAADV10,
    RunArtifacts as V10RunArtifacts,
)

from .._common import (
    ALL_TASKS,
    BACKENDS,
    EXPERIMENTS_ROOT,
    SAMPLING_TEMPERATURE,
    SAMPLING_TOP_K,
    SAMPLING_TOP_P,
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
    "v9_7",
    "v9_14",
    "v9_16",
    "v9_17",
    "v9_17_fixed_cycle",
    "v9_18_q_atomic",
    "v9_18_q_opportunity",
    "v9_18_facts",
    "v9_19",
    "v9_20",
    "v9_21",
    "v9_22",
    "v10",
]

VERSIONS: tuple[VersionName, ...] = (
    "v9_7",
    "v9_14",
    "v9_16",
    "v9_17",
    "v9_17_fixed_cycle",
    "v9_18_q_atomic",
    "v9_18_q_opportunity",
    "v9_18_facts",
    "v9_19",
    "v9_20",
    "v9_21",
    "v9_22",
    "v10",
)
TRACEAAD_V916_VERSIONS = {"v9_16"}
TRACEAAD_V917_VERSIONS = {"v9_17", "v9_17_fixed_cycle"}
TRACEAAD_V918_VERSIONS = {
    "v9_18_q_atomic",
    "v9_18_q_opportunity",
    "v9_18_facts",
}
TRACEAAD_V919_VERSIONS = {"v9_19"}
TRACEAAD_V920_VERSIONS = {"v9_20"}
TRACEAAD_V921_VERSIONS = {"v9_21"}
TRACEAAD_V922_VERSIONS = {"v9_22"}
TRACEAAD_V10_VERSIONS = {"v10"}


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
            else V916_INITIAL_ROOT_COUNT
            if version in TRACEAAD_V916_VERSIONS
            else V918_INITIAL_ROOT_COUNT
            if version in TRACEAAD_V918_VERSIONS
            else V919_INITIAL_ROOT_COUNT
            if version in TRACEAAD_V919_VERSIONS
            else V920_INITIAL_ROOT_COUNT
            if version in TRACEAAD_V920_VERSIONS
            else V921_INITIAL_ROOT_COUNT
            if version in TRACEAAD_V921_VERSIONS
            else V922_INITIAL_ROOT_COUNT
            if version in TRACEAAD_V922_VERSIONS
            else V10_INITIAL_ROOT_COUNT
            if version in TRACEAAD_V10_VERSIONS
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
                "v9_16",
                "v9_17",
                "v9_17_fixed_cycle",
                "v9_18_q_atomic",
                "v9_18_q_opportunity",
                "v9_18_facts",
                "v9_19",
                "v9_20",
                "v9_21",
                "v9_22",
                "v10",
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
    if spec.version in TRACEAAD_V916_VERSIONS and spec.n_init != V916_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.16 requires exactly eight initial roots")
    if spec.version in TRACEAAD_V917_VERSIONS and spec.n_init != V917_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.17 requires exactly eight initial roots")
    if spec.version in TRACEAAD_V918_VERSIONS and spec.n_init != V918_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.18 requires exactly eight initial roots")
    if spec.version in TRACEAAD_V919_VERSIONS and spec.n_init != V919_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.19 requires exactly eight initial roots")
    if spec.version in TRACEAAD_V920_VERSIONS and spec.n_init != V920_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.20 requires exactly eight initial roots")
    if spec.version in TRACEAAD_V921_VERSIONS and spec.n_init != V921_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.21 requires exactly eight initial roots")
    if spec.version in TRACEAAD_V922_VERSIONS and spec.n_init != V922_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V9.22 requires exactly eight initial roots")
    if spec.version in TRACEAAD_V10_VERSIONS and spec.n_init != V10_INITIAL_ROOT_COUNT:
        raise ValueError("TraceAAD V10 requires exactly eight initial roots")
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
    if spec.initialization_checkpoint is not None and spec.version not in (
        {"v9_17_fixed_cycle"} | TRACEAAD_V918_VERSIONS
    ):
        raise ValueError(
            "initialization_checkpoint is only valid for V9.17 FixedCycle or V9.18"
        )
    return spec


def build_method(
    spec: RunSpec,
    run_dir: Path,
    resume_from: Path | None = None,
):
    evaluation, task_kwargs = build_task(spec.task, spec.eval_workers)
    if spec.version in TRACEAAD_V919_VERSIONS:
        # V9.19 measures behavior on the training instances themselves: the
        # tracked evaluation returns the benchmark fitness plus trajectories.
        evaluation = V919_TRACKED_EVALUATIONS[spec.task](**task_kwargs)
    elif spec.version in TRACEAAD_V920_VERSIONS:
        evaluation = V920_TRACKED_EVALUATIONS[spec.task](**task_kwargs)
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
    if spec.version in TRACEAAD_V918_VERSIONS:
        allocation_mode, explore_context = _v918_modes(spec)
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
            allocation_mode=allocation_mode,
            explore_context=explore_context,
            fork_from_initialization=fork_from_initialization,
        )
    if spec.version in TRACEAAD_V919_VERSIONS:
        return TraceAADV919(
            llm=llm,
            evaluation=evaluation,
            artifacts=V919RunArtifacts(run_dir=run_dir),
            budget=spec.budget,
            n_roots=spec.n_init,
            max_tokens=spec.llm_output_tokens,
            seed=spec.seed,
            resume_from=resume_from,
            checkpoint_dir=run_dir / "checkpoints",
        )
    if spec.version in TRACEAAD_V920_VERSIONS:
        return TraceAADV920(
            llm=llm,
            evaluation=evaluation,
            artifacts=V920RunArtifacts(run_dir=run_dir),
            budget=spec.budget,
            n_roots=spec.n_init,
            max_tokens=spec.llm_output_tokens,
            seed=spec.seed,
            resume_from=resume_from,
            checkpoint_dir=run_dir / "checkpoints",
        )
    if spec.version in TRACEAAD_V921_VERSIONS:
        return TraceAADV921(
            llm=llm,
            evaluation=evaluation,
            artifacts=V921RunArtifacts(run_dir=run_dir),
            budget=spec.budget,
            n_roots=spec.n_init,
            max_tokens=spec.llm_output_tokens,
            seed=spec.seed,
            resume_from=resume_from,
            checkpoint_dir=run_dir / "checkpoints",
            task_key=spec.task,
        )
    if spec.version in TRACEAAD_V922_VERSIONS:
        return TraceAADV922(
            llm=llm,
            evaluation=evaluation,
            artifacts=V922RunArtifacts(run_dir=run_dir),
            budget=spec.budget,
            n_roots=spec.n_init,
            max_tokens=spec.llm_output_tokens,
            seed=spec.seed,
            resume_from=resume_from,
            checkpoint_dir=run_dir / "checkpoints",
            task_key=spec.task,
        )
    if spec.version in TRACEAAD_V10_VERSIONS:
        return TraceAADV10(
            llm=llm,
            evaluation=evaluation,
            artifacts=V10RunArtifacts(run_dir=run_dir),
            budget=spec.budget,
            n_roots=spec.n_init,
            max_tokens=spec.llm_output_tokens,
            seed=spec.seed,
            resume_from=resume_from,
            checkpoint_dir=run_dir / "checkpoints",
            task_key=spec.task,
            context_token_limit=spec.context_token_limit,
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
    elif spec.version in TRACEAAD_V918_VERSIONS:
        expected_method_params = _v918_method_params(spec)
    elif spec.version in TRACEAAD_V919_VERSIONS:
        expected_method_params = _v919_method_params(spec)
    elif spec.version in TRACEAAD_V920_VERSIONS:
        expected_method_params = _v920_method_params(spec)
    elif spec.version in TRACEAAD_V921_VERSIONS:
        expected_method_params = _v921_method_params(spec)
    elif spec.version in TRACEAAD_V922_VERSIONS:
        expected_method_params = _v922_method_params(spec)
    elif spec.version in TRACEAAD_V10_VERSIONS:
        expected_method_params = _v10_method_params(spec)
    else:
        raise ValueError(f"unsupported TraceAAD version: {spec.version}")
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
    elif spec.version == "v9_16":
        method_params = _v916_method_params(spec)
    elif spec.version in TRACEAAD_V918_VERSIONS:
        method_params = _v918_method_params(spec)
    elif spec.version in TRACEAAD_V919_VERSIONS:
        method_params = _v919_method_params(spec)
    elif spec.version in TRACEAAD_V920_VERSIONS:
        method_params = _v920_method_params(spec)
    elif spec.version in TRACEAAD_V921_VERSIONS:
        method_params = _v921_method_params(spec)
    elif spec.version in TRACEAAD_V922_VERSIONS:
        method_params = _v922_method_params(spec)
    elif spec.version in TRACEAAD_V10_VERSIONS:
        method_params = _v10_method_params(spec)
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
    # V9.18 artifacts need explicit service metadata for held-out provenance;
    # retain the established config shape of earlier TraceAAD versions.
    if spec.version in (
        TRACEAAD_V918_VERSIONS
        | TRACEAAD_V919_VERSIONS
        | TRACEAAD_V920_VERSIONS
        | TRACEAAD_V921_VERSIONS
        | TRACEAAD_V922_VERSIONS
        | TRACEAAD_V10_VERSIONS
    ):
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
        TRACEAAD_V916_VERSIONS
        | TRACEAAD_V917_VERSIONS
        | TRACEAAD_V918_VERSIONS
        | TRACEAAD_V919_VERSIONS
        | TRACEAAD_V920_VERSIONS
        | TRACEAAD_V921_VERSIONS
        | TRACEAAD_V922_VERSIONS
        | TRACEAAD_V10_VERSIONS
    ):
        return "Qwen3.6-27B"
    return V97_LOGICAL_MODEL_NAME


def _versioned_generator_environment(spec: RunSpec) -> dict[str, object]:
    return {
        "logical_model_name": _versioned_logical_model_name(spec),
        "temperature": SAMPLING_TEMPERATURE,
        "top_p": SAMPLING_TOP_P,
        "top_k": SAMPLING_TOP_K,
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


def _v918_modes(spec: RunSpec) -> tuple[str, str]:
    if spec.version == "v9_18_q_atomic":
        return "q", "legacy"
    if spec.version == "v9_18_q_opportunity":
        return "opportunity", "legacy"
    if spec.version == "v9_18_facts":
        return "q", "facts"
    raise ValueError(f"unsupported V9.18 mode: {spec.version}")


def _v918_method_params(spec: RunSpec) -> dict[str, object]:
    allocation_mode, explore_context = _v918_modes(spec)
    return {
        "budget": spec.budget,
        "n_roots": spec.n_init,
        "max_history": V918_MAX_HISTORY_EVENTS,
        "maximize": True,
        "max_tokens": spec.llm_output_tokens,
        "seed": spec.seed,
        "refine_probability": V918_REFINE_PROBABILITY,
        "explore_probability": V918_EXPLORE_PROBABILITY,
        "allocation_mode": allocation_mode,
        "explore_context": explore_context,
        "opportunity_lambda": V918_OPPORTUNITY_LAMBDA,
        "opportunity_tau": V918_OPPORTUNITY_TAU,
        "global_facts_window": V918_GLOBAL_FACTS_WINDOW,
        "ess_fraction": V918_ESS_FRACTION,
        "min_ess_target": V918_MIN_ESS_TARGET,
        "parent_score": (
            "quality_only"
            if allocation_mode == "q"
            else "q_plus_decaying_explore_entry_opportunity"
        ),
        "error_handling": True,
        "error_retries": 2,
        "retry_policy": "two_bounded_repairs",
        "retry_budget": "initial_candidates",
    }


def _v919_method_params(spec: RunSpec) -> dict[str, object]:
    return {
        "budget": spec.budget,
        "n_roots": spec.n_init,
        "max_history": V919_MAX_HISTORY_EVENTS,
        "maximize": True,
        "max_tokens": spec.llm_output_tokens,
        "seed": spec.seed,
        "mechanism": "behavior_grounded_formation_search",
        "node_score_weights": {
            "promise": V919_W_PROMISE,
            "underdevelopment": V919_W_UNDERDEVELOPMENT,
            "trajectory": V919_W_TRAJECTORY,
        },
        "trajectory_window": V919_TRAJECTORY_WINDOW,
        "explore_probability": {
            "neutral": V919_EXPLORE_NEUTRAL,
            "min": V919_EXPLORE_MIN,
            "max": V919_EXPLORE_MAX,
        },
        "actions": ["develop", "explore", "crossover"],
        "crossover_probability": V919_CROSSOVER_PROBABILITY,
        "behave_protocol": V919_BEHAVESIM_PROTOCOL_ID,
        "error_handling": True,
        "error_retries": V919_MAX_REPAIRS,
        "retry_policy": "two_bounded_repairs",
        "retry_budget": "initial_candidates",
    }


def _v920_method_params(spec: RunSpec) -> dict[str, object]:
    return {
        "budget": spec.budget,
        "n_roots": spec.n_init,
        "max_history": V920_MAX_HISTORY_EVENTS,
        "maximize": True,
        "max_tokens": spec.llm_output_tokens,
        "seed": spec.seed,
        "mechanism": "opportunity_allocation_plus_action_matched_assistance",
        "allocation_policy": "quality_continuation_plus_behavior_coverage",
        "coverage_mix": V920_COVERAGE_MIX,
        "ess_fraction": V920_ESS_FRACTION,
        "min_ess_target": V920_MIN_ESS_TARGET,
        "actions": ["develop", "explore", "crossover"],
        "action_temperature": V920_ACTION_TEMPERATURE,
        "explore_probability": {
            "neutral": V920_EXPLORE_NEUTRAL,
            "min": V920_EXPLORE_MIN,
            "max": V920_EXPLORE_MAX,
        },
        "behave_protocol": V920_BEHAVESIM_PROTOCOL_ID,
        "error_handling": True,
        "error_retries": V920_MAX_REPAIRS,
        "retry_policy": "two_bounded_repairs",
        "retry_budget": "primary_candidates",
    }


def _v921_method_params(spec: RunSpec) -> dict[str, object]:
    return {
        "budget": spec.budget,
        "n_roots": spec.n_init,
        "max_history": V921_MAX_HISTORY_EVENTS,
        "maximize": True,
        "max_tokens": spec.llm_output_tokens,
        "seed": spec.seed,
        "mechanism": "idea_hypothesis_paired_realization",
        "allocation": "one_step_hypothesis_ucb",
        "proposals_per_batch": ["continue", "branch"],
        "ideas_per_batch": 2,
        "realizations_per_idea": V921_REALIZATIONS_PER_IDEA,
        "primary_slots_per_batch": V921_BATCH_SIZE,
        "public_memory": "one_strict_improvement_card",
        "online_behavesim": False,
        "error_handling": True,
        "error_retries": V921_MAX_REPAIRS,
        "retry_policy": "two_bounded_repairs",
        "retry_budget": "primary_candidates",
    }


def _v922_method_params(spec: RunSpec) -> dict[str, object]:
    return {
        "budget": spec.budget,
        "n_roots": spec.n_init,
        "max_history": V922_MAX_HISTORY_EVENTS,
        "maximize": True,
        "max_tokens": spec.llm_output_tokens,
        "seed": spec.seed,
        "mechanism": "rank_calibrated_dual_baseline_hypothesis_search",
        "allocation": "hypothesis_ucb_plus_action_ucb",
        "proposals": ["continue", "branch"],
        "ideas_per_batch": V922_IDEAS_PER_BATCH,
        "realizations_per_idea": V922_REALIZATIONS_PER_IDEA,
        "primary_slots_per_batch": V922_BATCH_SIZE,
        "quality_calibration": "dynamic_midrank_over_current_scaffolds",
        "response_baselines": ["working", "scaffold"],
        "branch_context": "scaffold_only",
        "public_memory": "one_strict_improvement_card",
        "online_behavesim": False,
        "error_handling": True,
        "error_retries": V922_MAX_REPAIRS,
        "retry_policy": "two_bounded_repairs",
        "retry_budget": "primary_candidates",
    }


def _v10_method_params(spec: RunSpec) -> dict[str, object]:
    return {
        "budget": spec.budget,
        "n_roots": spec.n_init,
        "maximize": True,
        "max_tokens": spec.llm_output_tokens,
        "seed": spec.seed,
        "mechanism": "trajectory_aware_joint_design_opportunity_allocation",
        "allocation": "critic_competitive_set_plus_lexicographic_coverage",
        "operators": ["develop", "pivot", "transfer", "restart", "semantic_repair"],
        "K_s": V10_SCREEN_SIZE,
        "K_d": V10_REFERENCE_COUNT,
        "K_c": V10_COMPETITIVE_SET_SIZE,
        "H_tau": V10_FORMATION_WINDOW,
        "H_G": list(V10_G_HORIZONS),
        "N_card": V10_RESTART_CARDS,
        "critic_calls": "once_per_primary_slot_after_initialization",
        "generation_context": {
            "default": ["task", "current_algorithm", "formation_path", "operator_instruction"],
            "transfer": ["+reference_code", "+reference_formation_path"],
            "semantic_repair": ["+semantic_mismatch"],
            "restart": ["verified_improvement_cards_only"],
        },
        "error_handling": True,
        "error_retries": V10_MAX_REPAIRS,
        "retry_policy": "two_bounded_repairs",
        "retry_budget": "primary_candidates",
    }


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
    parser.add_argument("--experiments-root", type=Path, default=EXPERIMENTS_ROOT)
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
        experiments_root=args.experiments_root,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_experiment(spec_from_args(args))


if __name__ == "__main__":
    main()

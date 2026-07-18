"""Resume TraceAAD from the latest checkpoint under a run log directory."""
from __future__ import annotations

import copy
from pathlib import Path

from ...base import TextFunctionProgramConverter as tfpc
from .checkpoint import find_latest_checkpoint, load_checkpoint
from .traceaad import TraceAAD


def _resume_profiler(method: TraceAAD) -> None:
    profiler = method._profiler
    if profiler is None:
        return
    profiler._num_samples = int(method._tot_sample_nums)
    best = method._best_node
    if best is None or best.fitness is None:
        return
    function = tfpc.program_to_function(best.code)
    if function is None:
        template = copy.deepcopy(method._function_to_evolve)
        template.body = "    pass"
        template.score = best.fitness
        template.algorithm = best.idea
        function = template
    else:
        function.score = best.fitness
        function.algorithm = best.idea
    # Restore best display state without rewriting sample artifacts.
    if getattr(profiler, "_num_objs", 1) < 2:
        profiler._cur_best_function = function
        profiler._cur_best_program_score = float(best.fitness)
        profiler._cur_best_program_sample_order = int(method._tot_sample_nums)


def resume_traceaad(method: TraceAAD, log_dir: str | Path | None = None) -> Path:
    """Load the latest checkpoint into ``method`` and mark resume mode.

    ``log_dir`` defaults to ``method._profiler._log_dir``.
    Returns the checkpoint path that was loaded.
    """
    resolved = log_dir
    if resolved is None:
        profiler = method._profiler
        if profiler is None or not getattr(profiler, "_log_dir", None):
            raise ValueError("resume_traceaad requires log_dir or a profiler with _log_dir")
        resolved = profiler._log_dir
    ckpt = find_latest_checkpoint(resolved)
    load_checkpoint(method, ckpt)
    _resume_profiler(method)
    method._resume_mode = True
    if len(method._memory.active()) == 0:
        raise RuntimeError(
            f"checkpoint {ckpt} has no active trajectories; cannot resume search"
        )
    print(
        f"RESUME TraceAAD: checkpoint={ckpt} sample_order={method._tot_sample_nums} "
        f"active={len(method._memory.active())} "
        f"best={None if method._best_node is None else method._best_node.fitness}",
        flush=True,
    )
    return ckpt

# Module Name: FSSPGLSEvaluation
# Description: Evaluates guided local search perturbation heuristics for the
#              Flow Shop Scheduling Problem.
#
# Parameters:
#   - timeout_seconds: Maximum allowed time for program evaluation.
#   - split: Fixed dataset split used for evaluation.
#   - time_max: Maximum GLS running time per instance.
#   - iter_max: Maximum GLS iterations per instance.

from __future__ import annotations

import random
import time
from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.instance_parallel import evaluate_instances, validate_backend
from llm4ad.task.optimization.main.fssp_gls.dataset import (
    DEFAULT_SPLIT,
    load_split_instances,
)
from llm4ad.task.optimization.main.fssp_gls.template import template_program, task_description

__all__ = ["FSSPGLSEvaluation"]


def makespan(order: list[int], tasks: np.ndarray, machines_val: int) -> float:
    times = [0.0] * machines_val
    for job_idx in order:
        times[0] += tasks[job_idx][0]
        for machine_idx in range(1, machines_val):
            if times[machine_idx] < times[machine_idx - 1]:
                times[machine_idx] = times[machine_idx - 1]
            times[machine_idx] += tasks[job_idx][machine_idx]
    return max(times)


def local_search(sequence: list[int], cmax_old: float, tasks: np.ndarray, machines_val: int) -> list[int]:
    new_seq = sequence[:]
    for i in range(len(new_seq)):
        for j in range(i + 1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    for i in range(1, len(new_seq)):
        for j in range(1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq.remove(i)
            temp_seq.insert(j, i)
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    return new_seq


def local_search_perturb(
        sequence: list[int],
        cmax_old: float,
        tasks: np.ndarray,
        machines_val: int,
        jobs: list[int],
) -> list[int]:
    new_seq = sequence[:]
    for i in jobs:
        for j in range(i + 1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    for i in jobs:
        for j in range(1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq.remove(i)
            temp_seq.insert(j, i)
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax

    return new_seq


def sum_and_order(tasks_val: int, machines_val: int, tasks: np.ndarray) -> list[int]:
    job_loads = [0.0] * tasks_val
    order = [0] * tasks_val
    for job_idx in range(tasks_val):
        for machine_idx in range(machines_val):
            job_loads[job_idx] += tasks[job_idx][machine_idx]

    for position in range(tasks_val):
        max_time = -1.0
        max_job = 0
        for job_idx in range(tasks_val):
            if max_time < job_loads[job_idx]:
                max_time = job_loads[job_idx]
                max_job = job_idx
        job_loads[max_job] = -1.0
        order[position] = max_job
    return order


def neh(tasks: np.ndarray, machines_val: int, tasks_val: int) -> tuple[list[int], float]:
    order = sum_and_order(tasks_val, machines_val, tasks)
    current_seq = [order[0]]
    for i in range(1, tasks_val):
        min_cmax = float("inf")
        best_seq = None
        for j in range(0, i + 1):
            candidate = current_seq[:]
            candidate.insert(j, order[i])
            cmax = makespan(candidate, tasks, machines_val)
            if min_cmax > cmax:
                best_seq = candidate
                min_cmax = cmax
        current_seq = best_seq
    return current_seq, makespan(current_seq, tasks, machines_val)


def guided_local_search(
        tasks_val: int,
        tasks: np.ndarray,
        machines_val: int,
        time_max: float,
        iter_max: int,
        heuristic: Callable,
) -> float:
    cmax_best = 1e10
    random.seed(2024)
    try:
        pi, cmax = neh(tasks, machines_val, tasks_val)
        n = len(pi)
        cmax_best = cmax
        n_itr = 0
        time_start = time.time()
        while time.time() - time_start < time_max and n_itr < iter_max:
            pi = local_search(pi, cmax, tasks, machines_val)
            cmax = makespan(pi, tasks, machines_val)

            if cmax < cmax_best:
                cmax_best = cmax

            tasks_perturb, jobs = heuristic(pi, tasks.copy(), machines_val, n)
            tasks_perturb = np.asarray(tasks_perturb, dtype=float)
            jobs = [int(job) for job in list(jobs)]

            if tasks_perturb.shape != tasks.shape:
                return 1e10
            if len(jobs) <= 1:
                return 1e10
            jobs = [job for job in jobs if 0 <= job < n]
            if len(jobs) <= 1:
                return 1e10
            if len(jobs) > 5:
                jobs = jobs[:5]

            cmax = makespan(pi, tasks_perturb, machines_val)
            pi = local_search_perturb(pi, cmax, tasks_perturb, machines_val, jobs)

            n_itr += 1
            if n_itr % 50 == 0:
                cmax = cmax_best

    except Exception:
        cmax_best = 1e10

    return float(cmax_best)


def _evaluate_fssp_gls_instance(heuristic: Callable, instance, context) -> float:
    tasks = np.asarray(instance["processing_times"], dtype=float)
    return guided_local_search(
        tasks_val=int(instance["n_jobs"]),
        tasks=tasks,
        machines_val=int(instance["n_machines"]),
        time_max=context["time_max"],
        iter_max=context["iter_max"],
        heuristic=heuristic,
    )


class FSSPGLSEvaluation(Evaluation):
    """Evaluator for FSSP guided local search perturbation heuristics."""

    def __init__(
            self,
            timeout_seconds=120,
            split: str = DEFAULT_SPLIT,
            time_max: float = 30.0,
            iter_max: int = 1000,
            eval_workers: int = 1,
            eval_backend: str = "sequential",
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._datasets, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.time_max = float(time_max)
        self.iter_max = int(iter_max)
        self.eval_workers = max(1, int(eval_workers))
        self.eval_backend = validate_backend(eval_backend, daemon_eval_process=self.daemon_eval_process)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        try:
            makespans = evaluate_instances(
                program_str=program_str,
                callable_func=callable_func,
                payloads=list(self._datasets),
                instance_eval=_evaluate_fssp_gls_instance,
                context={
                    "time_max": self.time_max,
                    "iter_max": self.iter_max,
                },
                backend=self.eval_backend,
                workers=self.eval_workers,
                timeout_seconds=self.timeout_seconds,
            )
            return -float(np.mean(makespans))
        except Exception:
            return None

    def evaluate(self, heuristic: Callable) -> float:
        makespans = []
        for instance in self._datasets:
            tasks = np.asarray(instance["processing_times"], dtype=float)
            makespans.append(guided_local_search(
                tasks_val=int(instance["n_jobs"]),
                tasks=tasks,
                machines_val=int(instance["n_machines"]),
                time_max=self.time_max,
                iter_max=self.iter_max,
                heuristic=heuristic,
            ))
        return -float(np.mean(makespans))

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.instance_parallel import evaluate_instances, validate_backend
from llm4ad.task.optimization.main.bp_online.dataset import load_split_instances
from llm4ad.task.optimization.main.bp_online.template import task_description, template_program
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT

__all__ = ["BPOnlineEvaluation"]


def _get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    return np.nonzero((bins - item) >= 0)[0]


def _online_binpack(
        items: np.ndarray,
        bins: np.ndarray,
        score_fn: Callable[[int, np.ndarray], np.ndarray],
) -> np.ndarray | None:
    for item in items:
        valid = _get_valid_bin_indices(float(item), bins)
        if len(valid) == 0:
            return None
        priorities = score_fn(int(item), bins[valid].copy())
        priorities = np.asarray(priorities, dtype=float).flatten()
        if len(priorities) != len(valid) or not np.all(np.isfinite(priorities)):
            return None
        best = valid[int(np.argmax(priorities))]
        bins[best] -= item
    return bins


def _evaluate_bp_online_instance(score_fn: Callable, instance, context):
    capacity = int(instance["capacity"])
    items = np.asarray(instance["items"], dtype=int)
    bins = np.full(int(instance["num_items"]), capacity, dtype=float)
    bins_packed = _online_binpack(items, bins, score_fn)
    if bins_packed is None:
        return None
    used = int((bins_packed != capacity).sum())
    return instance["group_label"], used, float(instance["l1_bound"])


class BPOnlineEvaluation(Evaluation):
    """Evaluator for EoH's online bin-packing scoring task."""

    def __init__(
            self,
            timeout_seconds=40,
            split: str = DEFAULT_SPLIT,
            eval_workers: int = 1,
            eval_backend: str = "sequential",
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._instances, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.eval_workers = max(1, int(eval_workers))
        self.eval_backend = validate_backend(eval_backend, daemon_eval_process=self.daemon_eval_process)

    def get_valid_bin_indices(self, item: float, bins: np.ndarray) -> np.ndarray:
        return _get_valid_bin_indices(item, bins)

    def online_binpack(
            self,
            items: np.ndarray,
            bins: np.ndarray,
            score_fn: Callable[[int, np.ndarray], np.ndarray],
    ) -> np.ndarray | None:
        return _online_binpack(items, bins, score_fn)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        try:
            results = evaluate_instances(
                program_str=program_str,
                callable_func=callable_func,
                payloads=list(self._instances),
                instance_eval=_evaluate_bp_online_instance,
                backend=self.eval_backend,
                workers=self.eval_workers,
                timeout_seconds=self.timeout_seconds,
            )
            if any(result is None for result in results):
                return None

            used_by_group: dict[str, list[int]] = defaultdict(list)
            lb_by_group: dict[str, list[float]] = defaultdict(list)
            for group_label, used, l1_bound in results:
                used_by_group[group_label].append(used)
                lb_by_group[group_label].append(l1_bound)

            excess = []
            for group_label in sorted(used_by_group):
                avg_used = float(np.mean(used_by_group[group_label]))
                avg_lb = float(np.mean(lb_by_group[group_label]))
                excess.append((avg_used - avg_lb) / avg_lb)
            return -float(np.mean(excess))
        except Exception:
            return None

    def evaluate(self, score_fn: Callable[[int, np.ndarray], np.ndarray]) -> float | None:
        try:
            used_by_group: dict[str, list[int]] = defaultdict(list)
            lb_by_group: dict[str, list[float]] = defaultdict(list)

            for instance in self._instances:
                capacity = int(instance["capacity"])
                items = np.asarray(instance["items"], dtype=int)
                bins = np.full(int(instance["num_items"]), capacity, dtype=float)
                bins_packed = self.online_binpack(items, bins, score_fn)
                if bins_packed is None:
                    return None
                used_by_group[instance["group_label"]].append(int((bins_packed != capacity).sum()))
                lb_by_group[instance["group_label"]].append(float(instance["l1_bound"]))

            excess = []
            for group_label in sorted(used_by_group):
                avg_used = float(np.mean(used_by_group[group_label]))
                avg_lb = float(np.mean(lb_by_group[group_label]))
                excess.append((avg_used - avg_lb) / avg_lb)
            return -float(np.mean(excess))
        except Exception:
            return None

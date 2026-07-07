from __future__ import annotations

from typing import Any, Callable

import numpy as np

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.other.portfolio_construct.dataset import load_split_instances
from llm4ad.task.optimization.other.portfolio_construct.template import task_description, template_program

__all__ = ["PortfolioConstructEvaluation", "portfolio_sharpe"]


def portfolio_sharpe(asset_returns: np.ndarray, selected: np.ndarray) -> float:
    portfolio_returns = asset_returns[selected].mean(axis=0)
    mean = portfolio_returns.mean()
    std = portfolio_returns.std()
    if std < 1e-10:
        return 0.0
    return float(mean / std * np.sqrt(252))


class PortfolioConstructEvaluation(Evaluation):
    """Evaluator for EoH's greedy portfolio construction task."""

    def __init__(
            self,
            timeout_seconds=40,
            split: str = DEFAULT_SPLIT,
            n_select: int | None = None,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._instances, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.n_assets = int(self.dataset_metadata["n_assets"])
        self.n_periods = int(self.dataset_metadata["n_periods"])
        self.n_select = int(n_select if n_select is not None else self.dataset_metadata["n_select"])

    def _greedy_select(
            self,
            asset_returns: np.ndarray,
            score_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> np.ndarray | None:
        candidates = list(range(asset_returns.shape[0]))
        selected = []

        for _ in range(self.n_select):
            if not candidates:
                break
            candidate_indices = np.array(candidates, dtype=int)
            selected_indices = np.array(selected, dtype=int)
            scores = score_fn(asset_returns, selected_indices, candidate_indices)
            scores = np.asarray(scores, dtype=float).flatten()
            if len(scores) != len(candidates) or not np.all(np.isfinite(scores)):
                return None

            best = int(np.argmax(scores))
            selected.append(candidates[best])
            candidates.pop(best)

        return np.array(selected, dtype=int)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, score_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> float | None:
        try:
            sharpes = []
            for instance in self._instances:
                selected = self._greedy_select(instance["asset_returns"], score_fn)
                if selected is None or len(selected) < self.n_select:
                    return None
                sharpes.append(portfolio_sharpe(instance["asset_returns"], selected))
            return float(np.mean(sharpes))
        except Exception:
            return None

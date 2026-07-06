from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.cluster import KMeans

from llm4ad.base import Evaluation
from llm4ad.task.optimization.dataset_io import DEFAULT_SPLIT
from llm4ad.task.optimization.gnn_aggregation.dataset import load_split_instances
from llm4ad.task.optimization.gnn_aggregation.template import task_description, template_program

__all__ = ["GNNAggregationEvaluation"]


class GNNAggregationEvaluation(Evaluation):
    """Evaluator for EoH's GNN neighborhood-aggregation design task."""

    def __init__(
            self,
            timeout_seconds=40,
            split: str = DEFAULT_SPLIT,
            n_layers: int | None = None,
    ):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )

        self._instances, self.dataset_metadata = load_split_instances(split=split)
        self.n_instance = int(self.dataset_metadata["n_instances"])
        self.n_nodes = int(self.dataset_metadata["n_nodes"])
        self.n_feat = int(self.dataset_metadata["n_feat"])
        self.n_layers = int(n_layers if n_layers is not None else self.dataset_metadata["n_layers"])

    def _classification_error(self, features: np.ndarray, labels: np.ndarray) -> float:
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        pred = kmeans.fit_predict(features)
        acc = max(
            np.mean(pred == labels),
            np.mean(pred != labels),
        )
        return float(1.0 - acc)

    def evaluate_program(self, program_str: str, callable_func: Callable) -> Any | None:
        return self.evaluate(callable_func)

    def evaluate(self, aggregate_fn: Callable[[np.ndarray, np.ndarray, int], np.ndarray]) -> float | None:
        try:
            errors = []
            for instance in self._instances:
                adj = instance["adj_matrix"]
                labels = instance["labels"]
                feats = instance["node_features"].copy()

                for layer in range(self.n_layers):
                    feats = aggregate_fn(feats, adj, layer)
                    feats = np.asarray(feats, dtype=float)
                    if feats.ndim < 2 or feats.shape[0] != len(labels):
                        return None
                    if not np.all(np.isfinite(feats)):
                        return None

                errors.append(self._classification_error(feats, labels))

            return -float(np.mean(errors))
        except Exception:
            return None

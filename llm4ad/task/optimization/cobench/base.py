from __future__ import annotations

from typing import Any

from llm4ad.base import Evaluation
from llm4ad.task.optimization.cobench.utils import load_subdir_as_text

COBENCH_REPO_ID = "CO-Bench/CO-Bench"


def load_text_dataset(
        subdir: str,
        *,
        repo_id: str = COBENCH_REPO_ID,
) -> dict[str, str]:
    dataset = load_subdir_as_text(repo_id, subdir)
    return {
        filename: "\n".join(row["text"] for row in rows)
        for filename, rows in dataset.items()
    }


class COBenchEvaluation(Evaluation):
    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> Any | None:
        return self.evaluate(callable_func)

    def load_text_dataset(self, subdir: str) -> dict[str, str]:
        return load_text_dataset(subdir)

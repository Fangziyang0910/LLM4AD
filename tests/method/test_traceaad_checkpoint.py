from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad import TraceAAD, TraceAADProfiler

TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class CheckpointLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.draws = 0

    def draw_sample(self, prompt, *args, **kwargs):
        self.draws += 1
        if "next-step modifications" in prompt:
            return "1. Add a deterministic offset."
        return (
            f"Idea: candidate {self.draws}\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {self.draws}\n"
            "```"
        )


class CheckpointEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            use_numba_accelerate=False,
            safe_evaluate=False,
            timeout_seconds=10,
        )
        self.calls = 0

    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        return float(self.calls)


def _method(
    *,
    budget: int,
    checkpoint_dir: Path | None,
    resume_from: Path | None = None,
    llm: LLM | None = None,
    log_dir: Path | None = None,
) -> TraceAAD:
    return TraceAAD(
        llm=llm or CheckpointLLM(),
        evaluation=CheckpointEvaluation(),
        profiler=(
            None
            if log_dir is None
            else TraceAADProfiler(
                log_dir=str(log_dir),
                log_style="complex",
                create_random_path=False,
            )
        ),
        max_sample_nums=budget,
        n_init=2,
        actions_per_iteration=1,
        max_active_trajectories=10,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=2,
        resume_from=resume_from,
    )


def test_resume_continues_from_saved_budget_and_search_state(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    first = _method(budget=6, checkpoint_dir=checkpoint_dir).run()

    latest = checkpoint_dir / "latest.json"
    assert latest.is_file()
    assert first.n_samples == 6
    first_payload = json.loads(latest.read_text(encoding="utf-8"))
    assert set(first_payload["graph"]["nodes"][0]) == {
        "id",
        "code",
        "idea",
        "fitness",
    }
    assert "base_id" not in first_payload["memory"]["trajectories"][0]
    for trajectory in first_payload["memory"]["trajectories"]:
        if trajectory["value"] is not None:
            trajectory["value"]["diversity"] = 0.75
    latest.write_text(json.dumps(first_payload), encoding="utf-8")

    resumed = _method(
        budget=10,
        checkpoint_dir=None,
        resume_from=checkpoint_dir,
    ).run()

    assert resumed.n_samples == 10
    assert resumed.n_total_nodes > first.n_total_nodes
    assert resumed.n_edges >= first.n_edges
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["total_samples"] == 10


def test_interrupted_run_leaves_latest_checkpoint(tmp_path: Path) -> None:
    class InterruptingLLM(LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            raise KeyboardInterrupt

    checkpoint_dir = tmp_path / "checkpoints"
    method = _method(
        budget=6,
        checkpoint_dir=checkpoint_dir,
        llm=InterruptingLLM(),
    )

    with pytest.raises(KeyboardInterrupt):
        method.run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    assert payload["total_samples"] == 0
    assert payload["next_attempt_id"] == 0

    resumed = _method(
        budget=6,
        checkpoint_dir=None,
        resume_from=checkpoint_dir,
    ).run()
    assert resumed.n_samples == 6
    assert resumed.best_node is not None


def test_resume_rejects_a_different_search_configuration(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    _method(budget=2, checkpoint_dir=checkpoint_dir).run()

    with pytest.raises(ValueError, match="search configuration"):
        TraceAAD(
            llm=CheckpointLLM(),
            evaluation=CheckpointEvaluation(),
            max_sample_nums=4,
            n_init=2,
            actions_per_iteration=1,
            max_active_trajectories=10,
            softmax_temperature=0.3,
            resume_from=checkpoint_dir,
        )


def test_resume_continues_profiler_sample_numbers(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    checkpoint_dir = log_dir / "checkpoints"
    _method(
        budget=4,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
    ).run()

    resumed = _method(
        budget=6,
        checkpoint_dir=None,
        resume_from=checkpoint_dir,
        log_dir=log_dir,
    )
    resumed.run()

    assert resumed._profiler._num_samples == 6
    sample_orders = []
    for path in (log_dir / "samples").glob("samples_*.json"):
        if path.name == "samples_best.json":
            continue
        sample_orders.extend(
            row["sample_order"] for row in json.loads(path.read_text(encoding="utf-8"))
        )
    assert sorted(sample_orders) == [1, 2, 3, 4, 5, 6]

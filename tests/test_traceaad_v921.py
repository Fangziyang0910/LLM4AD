from __future__ import annotations

import csv
import json
from pathlib import Path

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_21 import RunArtifacts, TraceAADV921


class ToyEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program="def f(x):\n    return 0.0\n",
            task_description="Return a high scalar.",
            safe_evaluate=False,
            timeout_seconds=5,
        )

    def evaluate_program(self, program_str: str, callable_func, **kwargs):
        return float(callable_func(1.0))


class ToyLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.calls += 1
        if "do not write code" in prompt:
            return f"Idea: test direction {self.calls}"
        value = self.calls / 100.0
        return f"Idea: implementation {self.calls}\nCode:\n```python\ndef f(x):\n    return {value}\n```"


def test_v921_runs_paired_proposals_and_writes_evidence(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    method = TraceAADV921(
        ToyLLM(),
        ToyEvaluation(),
        RunArtifacts(run_dir),
        budget=12,
        n_roots=8,
        checkpoint_dir=run_dir / "checkpoints",
        task_key="toy",
    )
    method.run()

    assert method._n_eval == 12
    assert method._n_calls == 12
    assert len(method._hypotheses) == 9
    rows = list(csv.DictReader((run_dir / "evaluations.csv").open(newline="")))
    assert len(rows) == 12
    decisions = [json.loads(line) for line in (run_dir / "decisions.jsonl").read_text().splitlines()]
    assert {item["stage"] for item in decisions} == {"idea", "realization"}
    assert sum(item["stage"] == "idea" for item in decisions) == 2
    summary = json.loads((run_dir / "logs" / "summary.json").read_text())
    assert summary["mechanism"] == "hypothesis_search_paired_two_ideas_two_realizations"
    assert summary["online_behavesim"] is False
    assert (run_dir / "checkpoints" / "latest.json").is_file()


def test_v921_checkpoint_roundtrip(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    first = TraceAADV921(
        ToyLLM(), ToyEvaluation(), RunArtifacts(run_dir), budget=8, n_roots=8,
        checkpoint_dir=run_dir / "checkpoints", task_key="toy"
    )
    first.run()
    resumed = TraceAADV921(
        ToyLLM(), ToyEvaluation(), budget=8, n_roots=8,
        checkpoint_dir=tmp_path / "resumed" / "checkpoints",
        resume_from=run_dir / "checkpoints" / "latest.json", task_key="toy"
    )
    assert resumed._n_eval == 8
    assert resumed.best is not None
    assert len(resumed._hypotheses) == len(first._hypotheses)

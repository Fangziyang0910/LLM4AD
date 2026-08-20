from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.runners.traceaad import run
from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad_v9_7.prompt import (
    build_generation_prompt as build_v97_prompt,
    build_root_prompt as build_v97_root_prompt,
)
from llm4ad.method.traceaad_v9_7.traceaad import draw_intent as draw_intent_v97
from llm4ad.method.traceaad_v9_7_co import TraceAADV97CO
from llm4ad.method.traceaad_v9_7_co.prompt import (
    build_generation_prompt as build_co_prompt,
    build_root_prompt as build_co_root_prompt,
)
from llm4ad.method.traceaad_v9_7_co.traceaad import draw_intent as draw_intent_co

TEMPLATE = """def choose(value: int) -> int:
    return value
"""

HISTORY_TEXT = (
    "[Recent Algorithm Improvement History]\n\n"
    "[History 1] Formation step\nIdea: x\nChange: +1/-1 lines\n"
    "Result: improve\nFitness: 1 -> 1.25"
)


class ScriptedLLM(LLM):
    def __init__(self, start: int = 0) -> None:
        super().__init__()
        self.calls = start

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        return (
            f"Idea: candidate {self.calls}\nCode:\n```python\ndef choose(value: int) -> int:\n"
            f"    return value + {self.calls}\n```"
        )

    def count_tokens(self, text: str) -> int:
        return len(text) // 4 + 1


class ConstantEvaluation(Evaluation):
    def __init__(self) -> None:
        super().__init__(
            template_program=TEMPLATE,
            task_description="Improve choose.",
            safe_evaluate=False,
            timeout_seconds=10,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return 1.0


def test_co_prompt_is_v97_prompt_minus_the_history_block() -> None:
    shared = dict(
        task_description="Solve the task.",
        code="def f():\n    return 1",
        fitness=1.25,
        maximize=True,
    )
    for intent in ("refine", "explore"):
        from llm4ad.method.traceaad_v9_7.schema import Intent as V97Intent

        v97 = build_v97_prompt(
            history_text=HISTORY_TEXT, intent=V97Intent(intent), **shared
        )
        co = build_co_prompt(intent=V97Intent(intent), **shared)
        # everything outside the history block is byte-identical
        assert co == v97.replace(HISTORY_TEXT + "\n\n", "", 1).replace("\n\n\n", "\n\n", 1)
        assert "[Recent Algorithm Improvement History]" not in co
        assert "No history events" not in co
        assert "[Instruction]" in co and "[Current Algorithm]" in co


def test_co_root_prompt_is_byte_identical_to_v97() -> None:
    from llm4ad.base import TextFunctionProgramConverter

    template = TextFunctionProgramConverter.text_to_program(TEMPLATE)
    assert build_co_root_prompt(
        task_description="Improve choose.",
        template_function=template.functions[0],
        maximize=True,
    ) == build_v97_root_prompt(
        task_description="Improve choose.",
        template_function=template.functions[0],
        maximize=True,
    )


def test_co_intent_schedule_matches_v97() -> None:
    for seed in (None, 0, 1, 42):
        assert [draw_intent_co(seed, i) for i in range(500)] == [
            draw_intent_v97(seed, i) for i in range(500)
        ]


def test_co_smoke_run_and_events(tmp_path: Path) -> None:
    from llm4ad.method.traceaad_v9_7_co import RunArtifacts

    method = TraceAADV97CO(
        llm=ScriptedLLM(),
        evaluation=ConstantEvaluation(),
        artifacts=RunArtifacts(tmp_path, console_output=False),
        budget=30,
        context_limit=32768,
        checkpoint_dir=tmp_path / "checkpoints",
        seed=3,
    )
    method.run()
    summary = json.loads((tmp_path / "logs" / "summary.json").read_text())
    assert summary["status"] == "finished"
    assert summary["evaluator_call_count"] == 30
    assert summary["n_roots"] == 8
    # every recorded prompt is history-free
    import csv

    with (tmp_path / "evaluations.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(row["stage"] for row in rows)


def test_runner_builds_co_arm(tmp_path: Path) -> None:
    spec = run.make_run_spec(
        task="tsp_construct",
        version="v9_7_co",
        budget=1000,
        experiments_root=tmp_path,
    )
    method = run.build_method(spec, tmp_path / "run")
    assert isinstance(method, TraceAADV97CO)
    assert spec.method_name == "traceaad_v9_7_co"
    assert spec.n_init == 8
    assert spec.context_token_limit == 32768
    config = method.search_configuration()
    assert config["generation_context"] == "code_only"
    assert config == run._v97co_method_params(spec)
    with pytest.raises(ValueError, match="exactly eight"):
        run.make_run_spec(
            task="tsp_construct", version="v9_7_co", n_init=10, experiments_root=tmp_path
        )
    method._llm.close()

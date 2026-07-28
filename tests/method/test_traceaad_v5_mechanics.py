"""Public mechanism tests for the current TraceAAD V5.3 contract."""

from __future__ import annotations

import json
from pathlib import Path

from llm4ad.base import Evaluation, LLM


TEMPLATE = """def choose(value: int) -> int:
    return value
"""


class ScriptedV5LLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.prompts: list[str] = []

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        self.prompts.append(str(prompt))
        if "Generate a complete implementation" in prompt:
            return self._program(self.calls)
        if "[Requested Modification]" in prompt:
            return self._program(self.calls)
        return "1. Add one deterministic offset to the current rule."

    @staticmethod
    def _program(value: int) -> str:
        return (
            f"Idea: deterministic candidate {value}\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {value}\n"
            "```"
        )


class IncreasingEvaluation(Evaluation):
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


class V5MechanismLLM(ScriptedV5LLM):
    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        self.prompts.append(str(prompt))
        if "Generate a complete implementation" in prompt:
            return self._program(self.calls)
        if "[Requested Modification]" in prompt:
            return self._program(self.calls)
        return "1. Adapt one deterministic offset without adding branches."


class ParsimonyLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.program_draws = 0

    def draw_sample(self, prompt, *args, **kwargs):
        if "[Action Contract]" in prompt:
            return "1. Remove redundant assignments."
        self.program_draws += 1
        bodies = {
            1: "    copied = value\n    redundant = copied\n    return redundant\n",
            2: "    return value\n",
            3: "    return int(value)\n",
        }
        return (
            f"Idea: implementation {self.program_draws}\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            f"{bodies[self.program_draws]}"
            "```"
        )


class ConstantEvaluation(IncreasingEvaluation):
    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        return 1.0


class LongProgramLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.program_draws = 0

    def draw_sample(self, prompt, *args, **kwargs):
        if "[Action Contract]" in prompt:
            return "1. Implement a longer candidate with a genuinely better rule."
        self.program_draws += 1
        if self.program_draws == 1:
            body = "    return value\n"
        else:
            body = "".join(f"    workspace_{index} = value\n" for index in range(120))
            body += "    quality_marker = value + 1\n    return quality_marker\n"
        return (
            f"Idea: candidate {self.program_draws}\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            f"{body}"
            "```"
        )


class MarkerEvaluation(IncreasingEvaluation):
    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        return 2.0 if "quality_marker" in program_str else 1.0


class FirstEvaluationFails(IncreasingEvaluation):
    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.calls += 1
        return None if self.calls == 1 else float(self.calls)


class ExecutingEvaluation(Evaluation):
    TEMPLATE = """import numpy as np

def choose(value: int) -> int:
    return value
"""

    def __init__(self) -> None:
        super().__init__(
            template_program=self.TEMPLATE,
            task_description="Improve choose.",
            use_numba_accelerate=False,
            safe_evaluate=True,
            timeout_seconds=10,
        )

    def evaluate_program(self, program_str, callable_func, **kwargs):
        return float(callable_func(3))


class FunctionOnlyNumpyLLM(LLM):
    def draw_sample(self, prompt, *args, **kwargs):
        return (
            "Idea: use the template NumPy dependency\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            "    return int(np.asarray([value]).sum())\n"
            "```"
        )


class HelperFunctionLLM(LLM):
    def draw_sample(self, prompt, *args, **kwargs):
        return (
            "Idea: keep a small reusable helper\n"
            "```python\n"
            "def offset(value: int) -> int:\n"
            "    return value + 2\n\n"
            "def choose(value: int) -> int:\n"
            "    return offset(value)\n"
            "```"
        )


class RuntimeFailureLLM(LLM):
    def draw_sample(self, prompt, *args, **kwargs):
        return (
            "Idea: expose a generated runtime failure\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            "    raise ValueError('generated failure')\n"
            "```"
        )


class SingleActionLLM(ScriptedV5LLM):
    def draw_sample(self, prompt, *args, **kwargs):
        if "[Action Contract]" in prompt:
            return (
                "1. Try one explicit hypothesis even when only one action is returned."
            )
        return super().draw_sample(prompt, *args, **kwargs)


class NaturalLanguageActionLLM(ScriptedV5LLM):
    ACTIONS = (
        "Replace the fixed offset with a state-dependent adjustment, unlike the "
        "previous constant rule.",
        "Simplify the current rule by removing the redundant branch while preserving "
        "its useful behavior.",
    )

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        self.prompts.append(str(prompt))
        if "Generate a complete implementation" in prompt:
            return self._program(self.calls)
        if "[Requested Modification]" in prompt:
            return self._program(self.calls)
        return "\n".join(
            f"{index}. {action}" for index, action in enumerate(self.ACTIONS, start=1)
        )


class NoisyActionLLM(NaturalLanguageActionLLM):
    def draw_sample(self, prompt, *args, **kwargs):
        if "[Action Contract]" in prompt:
            self.calls += 1
            self.prompts.append(str(prompt))
            return (
                "Here are two actions:\n"
                f"1. {self.ACTIONS[0]}\n"
                "This first change can be implemented locally.\n"
                f"2. {self.ACTIONS[1]}"
            )
        return super().draw_sample(prompt, *args, **kwargs)


class MultipleCodeBlocksLLM(ScriptedV5LLM):
    ACTION = "Replace the current rule with the requested final local adjustment."

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        self.prompts.append(str(prompt))
        if "Generate a complete implementation" in prompt:
            return self._program(self.calls)
        if "[Requested Modification]" not in prompt:
            return f"1. {self.ACTION}"
        return (
            "Idea: use the final local adjustment\n"
            "An earlier sketch was:\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            "    return value + 100\n"
            "```\n"
            "The complete implementation is:\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            "    return value + 200\n"
        )


class VerboseIdeaLLM(ScriptedV5LLM):
    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        self.prompts.append(str(prompt))
        if "[Action Contract]" in prompt:
            return "1. Make one feasible local adjustment."
        return (
            "Idea: " + "descriptive implementation detail " * 30 + "\n"
            "```python\n"
            "def choose(value: int) -> int:\n"
            f"    return value + {self.calls}\n"
            "```"
        )


def test_traceaad_v5_has_an_independent_public_method_class() -> None:
    from llm4ad.method.traceaad import TraceAAD
    from llm4ad.method.traceaad_v5 import TraceAADV5

    assert TraceAADV5 is not TraceAAD
    assert TraceAADV5.__module__.startswith("llm4ad.method.traceaad_v5")


def test_traceaad_v5_runs_text_actions_and_writes_v5_state(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5

    checkpoint_dir = tmp_path / "checkpoints"
    result = TraceAADV5(
        llm=ScriptedV5LLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=4,
        n_init=2,
        actions_per_iteration=1,
        max_active_trajectories=4,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=1,
    ).run()

    assert result.n_samples == 4
    assert result.n_edges == 2
    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    assert "format_version" not in payload
    assert all(
        {"program_loc", "code_hash"} <= set(node) for node in payload["graph"]["nodes"]
    )
    assert all(
        len(route["node_ids"]) == len(route["edge_ids"]) + 1
        for route in payload["memory"]["trajectories"]
    )
    assert payload["graph"]["edges"][0]["action"].startswith("Add one")


def test_traceaad_v5_truncates_trajectories_to_the_configured_length(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    checkpoint_dir = tmp_path / "checkpoints"
    TraceAADV5(
        llm=V5MechanismLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=5,
        n_init=1,
        actions_per_iteration=1,
        max_trajectory_length=2,
        max_active_trajectories=1,
        operators=(TraceIdeateOp,),
        random_seed=3,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    routes = payload["memory"]["trajectories"]
    assert max(len(route["node_ids"]) for route in routes) == 2
    assert max(len(route["edge_ids"]) for route in routes) == 1
    assert all(len(route["node_ids"]) <= 2 for route in routes)
    assert all(route["compact_best_id"] in route["node_ids"] for route in routes)


def test_traceaad_v5_selects_an_anchor_inside_the_selected_trajectory() -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.derivation_graph import DerivationGraph
    from llm4ad.method.traceaad_v5.operators import TraceRefineOp
    from llm4ad.method.traceaad_v5.schema import Trajectory

    method = TraceAADV5(
        llm=V5MechanismLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=0,
        n_init=0,
        random_seed=3,
    )
    graph = DerivationGraph()
    compact = graph.add_node(
        code="def choose(value):\n    compact_anchor_marker = value\n    return value\n",
        idea="compact route best",
        fitness=2.0,
    )
    endpoint = graph.add_node(
        code="def choose(value):\n    endpoint_anchor_marker = value\n    return value\n",
        idea="later endpoint",
        fitness=1.0,
    )
    edge = graph.add_edge(
        parent_id=compact.id,
        child_id=endpoint.id,
        action="test a later change",
        operator="trace_refine",
        anchor_role="endpoint",
        primary_trajectory_id=0,
        delta_parent=-1.0,
        outcome="regress",
    )
    route = Trajectory(
        id=0,
        node_ids=(compact.id, endpoint.id),
        edge_ids=(edge.id,),
        endpoint_id=endpoint.id,
        compact_best_id=compact.id,
    )
    method._graph = graph

    assert method._max_context_tokens is None
    assert method._prompt_fits("long prompt " * 100_000)
    selected_anchors = {method._select_anchor(route) for _ in range(20)}
    assert selected_anchors == {
        (endpoint.id, "endpoint"),
        (compact.id, "compact_best"),
    }
    for anchor_id, marker in (
        (endpoint.id, "endpoint_anchor_marker"),
        (compact.id, "compact_anchor_marker"),
    ):
        context = method._build_action_context(
            selected=route,
            anchor_id=anchor_id,
            anchor_role="test",
            operator=TraceRefineOp(),
            reference_route=None,
            reference_node=None,
            log_result=False,
        )
        assert context is not None
        current_program = context.prompt.split("[Current Program]", maxsplit=1)[1]
        assert marker in current_program


def test_traceaad_v5_reference_is_provenance_not_a_second_parent(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceTransferOp

    llm = V5MechanismLLM()
    checkpoint_dir = tmp_path / "checkpoints"
    result = TraceAADV5(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        n_init=2,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceTransferOp,),
        random_seed=2,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    edge = payload["graph"]["edges"][0]
    assert result.n_edges == 1
    assert edge["reference_trajectory_id"] is not None
    assert edge["reference_program_id"] is not None
    assert (
        sum(
            candidate["child_id"] == edge["child_id"]
            for candidate in payload["graph"]["edges"]
        )
        == 1
    )
    assert any("[Reference Program]" in prompt for prompt in llm.prompts)


def test_traceaad_v53_has_no_online_global_experience(tmp_path: Path) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    llm = V5MechanismLLM()
    checkpoint_dir = tmp_path / "checkpoints"
    result = TraceAADV5(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=4,
        n_init=1,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceIdeateOp,),
        random_seed=4,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    removed_state = {
        "global_experience",
        "pending_reflection_edge_ids",
        "experience_reflection_attempts",
        "experience_update_index",
    }

    assert result.n_samples == 4
    assert removed_state.isdisjoint(payload)
    assert not hasattr(result, "n_experience_updates")
    assert all("[Global Experience]" not in prompt for prompt in llm.prompts)
    assert all("[Recent Code Changes]" not in prompt for prompt in llm.prompts)


def test_traceaad_v5_checkpoint_resumes_search_state(tmp_path: Path) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADProfiler, TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    checkpoint_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    first = TraceAADV5(
        llm=V5MechanismLLM(),
        evaluation=IncreasingEvaluation(),
        profiler=TraceAADProfiler(
            log_dir=str(log_dir), log_style="simple", create_random_path=False
        ),
        max_sample_nums=3,
        n_init=2,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceIdeateOp,),
        random_seed=7,
        checkpoint_dir=checkpoint_dir,
    ).run()
    resumed = TraceAADV5(
        llm=V5MechanismLLM(),
        evaluation=IncreasingEvaluation(),
        profiler=TraceAADProfiler(
            log_dir=str(log_dir), log_style="simple", create_random_path=False
        ),
        max_sample_nums=5,
        n_init=2,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceIdeateOp,),
        random_seed=999,
        resume_from=checkpoint_dir / "latest.json",
    ).run()

    assert first.n_samples == 3
    assert resumed.n_samples == 5
    assert resumed.n_total_nodes == first.n_total_nodes + 2
    summary = json.loads((log_dir / "run_summary.json").read_text())
    assert summary["num_samples"] == 5
    assert summary["evaluate_success_program_num"] == 5
    assert summary["total_sample_time"] > 0
    assert summary["llm_call_count"] == len(
        (log_dir / "llm_calls.jsonl").read_text().splitlines()
    )
    assert summary["method_event_count"] == len(
        (log_dir / "method_events.jsonl").read_text().splitlines()
    )


def test_traceaad_v5_writes_partial_summary_and_checkpoint_on_exception(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADProfiler, TraceAADV5

    class BrokenLLM(LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            raise RuntimeError("generation broke")

    log_dir = tmp_path / "logs"
    checkpoint = log_dir / "checkpoints" / "latest.json"
    method = TraceAADV5(
        llm=BrokenLLM(),
        evaluation=IncreasingEvaluation(),
        profiler=TraceAADProfiler(
            log_dir=str(log_dir), log_style="simple", create_random_path=False
        ),
        max_sample_nums=2,
        n_init=1,
        debug_mode=True,
        checkpoint_dir=checkpoint.parent,
    )

    try:
        method.run()
    except RuntimeError as exc:
        assert str(exc) == "generation broke"
    else:
        raise AssertionError("expected generation failure")

    summary = json.loads((log_dir / "run_summary.json").read_text())
    assert summary["status"] == "error"
    assert summary["error_type"] == "RuntimeError"
    assert summary["error"] == "generation broke"
    assert summary["num_samples"] == 0
    assert checkpoint.is_file()


def test_traceaad_v5_abort_during_init_keeps_initialization_incomplete(
    tmp_path: Path,
) -> None:
    """Init LLM failures must not freeze an empty search as initialization_complete."""

    class AlwaysFailLLM(LLM):
        def draw_sample(self, prompt, *args, **kwargs):
            raise RuntimeError("llm unavailable")

    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    checkpoint_dir = tmp_path / "checkpoints"
    aborted = TraceAADV5(
        llm=AlwaysFailLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=20,
        n_init=3,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceIdeateOp,),
        max_consecutive_sample_failures=3,
        random_seed=0,
        checkpoint_dir=checkpoint_dir,
    ).run()
    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))

    assert aborted.n_samples == 0
    assert payload["initialization_complete"] is False
    assert payload["total_samples"] == 0
    assert payload["memory"]["trajectories"] == []

    resumed = TraceAADV5(
        llm=V5MechanismLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=4,
        n_init=3,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceIdeateOp,),
        random_seed=1,
        resume_from=checkpoint_dir / "latest.json",
    ).run()
    assert resumed.n_samples == 4
    assert resumed.n_trajectories >= 3


def test_traceaad_v5_prefers_shorter_programs_on_exact_fitness_ties(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceRefineOp

    checkpoint_dir = tmp_path / "checkpoints"
    result = TraceAADV5(
        llm=ParsimonyLLM(),
        evaluation=ConstantEvaluation(),
        max_sample_nums=3,
        n_init=1,
        actions_per_iteration=1,
        max_active_trajectories=1,
        operators=(TraceRefineOp,),
        random_seed=0,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    assert result.best_node is not None
    assert result.best_node.id == 1
    assert result.best_node.program_loc == 2
    assert payload["graph"]["edges"][0]["global_best_update_reason"] == "tie_shorter"
    assert payload["graph"]["edges"][1]["new_global_best"] is False


def test_traceaad_v5_executes_an_available_text_action(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    checkpoint_dir = tmp_path / "checkpoints"
    result = TraceAADV5(
        llm=SingleActionLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=2,
        n_init=1,
        actions_per_iteration=2,
        max_active_trajectories=2,
        operators=(TraceIdeateOp,),
        random_seed=0,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    assert result.n_samples == 2
    assert result.n_edges == 1
    assert payload["graph"]["edges"][0]["action"] == (
        "Try one explicit hypothesis even when only one action is returned."
    )


def test_traceaad_v5_executes_natural_language_actions_without_metadata_fields(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    llm = NaturalLanguageActionLLM()
    checkpoint_dir = tmp_path / "checkpoints"
    result = TraceAADV5(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        n_init=1,
        actions_per_iteration=2,
        max_active_trajectories=2,
        operators=(TraceIdeateOp,),
        max_stalled_iterations=1,
        random_seed=0,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    action_prompts = [prompt for prompt in llm.prompts if "[Action Contract]" in prompt]
    code_prompts = [
        prompt for prompt in llm.prompts if "[Requested Modification]" in prompt
    ]

    assert result.n_samples == 3
    assert result.n_edges == 2
    assert len(action_prompts) == 1
    assert "numbered single-line actions" in action_prompts[0]
    assert [edge["action"] for edge in payload["graph"]["edges"]] == list(
        NaturalLanguageActionLLM.ACTIONS
    )
    assert all(
        edge["operator"] == "trace_ideate"
        and edge["primary_trajectory_id"] == 0
        and edge["anchor_role"] == "endpoint_compact_best"
        for edge in payload["graph"]["edges"]
    )
    assert all(
        any(action in prompt for prompt in code_prompts)
        for action in NaturalLanguageActionLLM.ACTIONS
    )


def test_traceaad_v5_extracts_only_numbered_actions_and_requests_feasible_changes(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    llm = NoisyActionLLM()
    checkpoint_dir = tmp_path / "checkpoints"
    result = TraceAADV5(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        n_init=1,
        actions_per_iteration=2,
        max_active_trajectories=2,
        operators=(TraceIdeateOp,),
        random_seed=0,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    action_prompt = next(
        prompt for prompt in llm.prompts if "[Action Contract]" in prompt
    )

    assert result.n_edges == 2
    assert [edge["action"] for edge in payload["graph"]["edges"]] == list(
        NoisyActionLLM.ACTIONS
    )
    assert "using only its arguments and locally computed values" in action_prompt
    assert (
        "Do not assume hidden state or change the function signature" in action_prompt
    )


def test_traceaad_v5_code_stage_only_implements_action_and_uses_final_program(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    llm = MultipleCodeBlocksLLM()
    checkpoint_dir = tmp_path / "checkpoints"
    result = TraceAADV5(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=2,
        n_init=1,
        actions_per_iteration=1,
        max_active_trajectories=2,
        operators=(TraceIdeateOp,),
        random_seed=0,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    code_prompt = next(
        prompt for prompt in llm.prompts if "[Requested Modification]" in prompt
    )
    child = payload["graph"]["nodes"][payload["graph"]["edges"][0]["child_id"]]

    assert result.n_edges == 1
    assert "return value + 200" in child["code"]
    assert "return value + 100" not in child["code"]
    assert MultipleCodeBlocksLLM.ACTION in code_prompt
    assert "[Operator Constraint]" not in code_prompt
    assert "Propose one genuinely new algorithmic idea" not in code_prompt


def test_traceaad_v5_prompt_stages_only_receive_decision_relevant_context(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceTransferOp

    llm = V5MechanismLLM()
    TraceAADV5(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        n_init=2,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceTransferOp,),
        random_seed=2,
        checkpoint_dir=tmp_path / "checkpoints",
    ).run()

    action_prompt = next(
        prompt for prompt in llm.prompts if "[Action Contract]" in prompt
    )
    code_prompt = next(
        prompt for prompt in llm.prompts if "[Requested Modification]" in prompt
    )

    assert "[Current Program]" in action_prompt
    assert "[Reference Program]" in action_prompt
    assert "[Improvement Direction]" in action_prompt
    assert "[Global Experience]" not in action_prompt
    assert all(
        internal not in action_prompt
        for internal in ("Node p", "anchor_role=", "knowledge provenance", "operator=")
    )
    assert "[Current Program]" in code_prompt
    assert "[Current Program History]" in code_prompt
    assert "[Reference Program History]" in code_prompt
    assert "[Reference Program]" in code_prompt
    assert "[Improvement Direction]" not in code_prompt
    assert "[Global Experience]" not in code_prompt
    assert "use it only in the way specified" in code_prompt
    assert "anchor_role" not in code_prompt
    assert "provenance" not in code_prompt
    assert "history is only used for faithful implementation" not in code_prompt.lower()


def test_traceaad_v5_history_distinguishes_planned_and_implemented_changes(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    llm = V5MechanismLLM()
    TraceAADV5(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        n_init=1,
        actions_per_iteration=1,
        max_active_trajectories=1,
        operators=(TraceIdeateOp,),
        random_seed=3,
        checkpoint_dir=tmp_path / "checkpoints",
    ).run()

    action_prompts = [prompt for prompt in llm.prompts if "[Action Contract]" in prompt]
    code_prompts = [
        prompt for prompt in llm.prompts if "[Requested Modification]" in prompt
    ]

    assert any("Planned:" in prompt for prompt in action_prompts[1:])
    assert any("Implemented:" in prompt for prompt in action_prompts[1:])
    assert any("Code change:" in prompt for prompt in action_prompts[1:])
    assert any("Planned:" in prompt for prompt in code_prompts[1:])
    assert any("Implemented:" in prompt for prompt in code_prompts[1:])
    assert any("Code change:" in prompt for prompt in code_prompts[1:])


def test_traceaad_v5_history_does_not_repeat_step_code() -> None:
    from llm4ad.method.traceaad_v5.context import trajectory_history
    from llm4ad.method.traceaad_v5.derivation_graph import DerivationGraph
    from llm4ad.method.traceaad_v5.trajectory_memory import TrajectoryMemory

    graph = DerivationGraph()
    parent = graph.add_node(
        code="def choose(value):\n    ancestor_code_marker = value\n    return value\n",
        idea="initial rule",
        fitness=1.0,
    )
    child = graph.add_node(
        code="def choose(value):\n    child_code_marker = value + 1\n    return value\n",
        idea="implemented a local adjustment",
        fitness=2.0,
    )
    memory = TrajectoryMemory(max_trajectory_length=8)
    initial = memory.create_initial(node_id=parent.id)
    edge = graph.add_edge(
        parent_id=parent.id,
        child_id=child.id,
        action="adjust the local rule",
        operator="trace_refine",
        anchor_role="endpoint",
        primary_trajectory_id=initial.id,
        delta_parent=1.0,
        outcome="improve",
    )
    route = memory.branch_from(
        trajectory_id=initial.id,
        base_node_id=parent.id,
        child_id=child.id,
        edge_id=edge.id,
        compact_best_id=child.id,
    )

    rendered = trajectory_history(graph, route, max_steps=8).text

    assert "Planned: adjust the local rule" in rendered
    assert "Implemented: implemented a local adjustment" in rendered
    assert "ancestor_code_marker" not in rendered
    assert "child_code_marker" not in rendered


def test_traceaad_v5_internal_anchor_receives_the_whole_retained_trajectory() -> None:
    from llm4ad.method.traceaad_v5.context import trajectory_history
    from llm4ad.method.traceaad_v5.derivation_graph import DerivationGraph
    from llm4ad.method.traceaad_v5.schema import Trajectory

    graph = DerivationGraph()
    nodes = [
        graph.add_node(
            code=f"def choose(value):\n    return value + {index}\n",
            idea=f"implementation {index}",
            fitness=float(index),
        )
        for index in range(8)
    ]
    edges = [
        graph.add_edge(
            parent_id=nodes[index].id,
            child_id=nodes[index + 1].id,
            action=f"change {index}",
            operator="trace_refine",
            anchor_role="endpoint",
            primary_trajectory_id=0,
            delta_parent=1.0,
            outcome="improve",
        )
        for index in range(7)
    ]
    route = Trajectory(
        id=0,
        node_ids=tuple(node.id for node in nodes),
        edge_ids=tuple(edge.id for edge in edges),
        endpoint_id=nodes[-1].id,
        compact_best_id=nodes[-1].id,
    )

    rendered = trajectory_history(
        graph,
        route,
        base_node_id=nodes[-2].id,
        max_steps=8,
    )

    assert rendered.edge_ids == tuple(edge.id for edge in edges)
    assert rendered.formation_edge_ids == tuple(edge.id for edge in edges[:6])
    assert rendered.tested_after_edge_ids == (edges[-1].id,)


def test_traceaad_v5_uses_focused_refine_and_task_grounded_action_guidance() -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceRefineOp

    refine = TraceRefineOp().build_constraint()
    assert "focused, evidence-grounded refinement" in refine
    assert all(
        operation not in refine
        for operation in ("replace", "delete", "merge", "simplify")
    )

    llm = V5MechanismLLM()
    TraceAADV5(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=2,
        n_init=1,
        actions_per_iteration=1,
        max_active_trajectories=1,
        operators=(TraceRefineOp,),
        random_seed=0,
    ).run()
    action_prompt = next(
        prompt for prompt in llm.prompts if "[Action Contract]" in prompt
    )
    assert "avoid constants justified only by the observed training size" in action_prompt


def test_traceaad_v5_search_value_blends_quality_and_trend() -> None:
    from llm4ad.method.traceaad_v5.derivation_graph import DerivationGraph
    from llm4ad.method.traceaad_v5.trajectory_memory import TrajectoryMemory
    from llm4ad.method.traceaad_v5.value import (
        ValueWeights,
        score_active_trajectories,
    )

    graph = DerivationGraph()
    parent = graph.add_node(code="def f():\n    return 1\n", idea="p", fitness=1.0)
    child = graph.add_node(code="def f():\n    return 2\n", idea="c", fitness=2.0)
    other = graph.add_node(code="def f():\n    return 3\n", idea="o", fitness=3.0)
    memory = TrajectoryMemory(max_trajectory_length=8)
    initial = memory.create_initial(node_id=parent.id)
    edge = graph.add_edge(
        parent_id=parent.id,
        child_id=child.id,
        action="improve",
        operator="trace_refine",
        anchor_role="endpoint",
        primary_trajectory_id=initial.id,
        delta_parent=1.0,
    )
    memory.branch_from(
        trajectory_id=initial.id,
        base_node_id=parent.id,
        child_id=child.id,
        edge_id=edge.id,
        compact_best_id=child.id,
    )
    memory.archive(initial.id)
    memory.create_initial(node_id=other.id)

    weights = ValueWeights()
    scored = score_active_trajectories(
        memory=memory,
        graph=graph,
        maximize=True,
        w=weights,
    )

    assert weights.search_quality == 0.8
    assert weights.search_trend == 0.2
    assert all(route.value is not None for route in scored)
    assert all(
        abs(
            float(route.scalar_value)
            - (0.8 * route.value.quality + 0.2 * route.value.trend)
        )
        < 1e-12
        for route in scored
        if route.value is not None and route.scalar_value is not None
    )


def test_traceaad_v5_keeps_idea_as_a_short_implementation_claim(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    llm = VerboseIdeaLLM()
    checkpoint_dir = tmp_path / "checkpoints"
    TraceAADV5(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=2,
        n_init=1,
        actions_per_iteration=1,
        max_active_trajectories=2,
        operators=(TraceIdeateOp,),
        random_seed=0,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    generation_prompts = [
        prompt
        for prompt in llm.prompts
        if "Generate a complete implementation" in prompt
        or "[Requested Modification]" in prompt
    ]

    assert all(len(node["idea"]) <= 300 for node in payload["graph"]["nodes"])
    assert all("no more than 300 characters" in prompt for prompt in generation_prompts)


def test_traceaad_v5_retries_failed_evaluations_until_n_init_is_reached(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5

    result = TraceAADV5(
        llm=ScriptedV5LLM(),
        evaluation=FirstEvaluationFails(),
        max_sample_nums=3,
        n_init=2,
        actions_per_iteration=1,
        max_active_trajectories=2,
        random_seed=0,
        checkpoint_dir=tmp_path / "checkpoints",
    ).run()

    assert result.n_samples == 3
    assert result.n_valid_nodes == 2
    assert result.n_trajectories == 2
    assert result.n_edges == 0


def test_traceaad_v5_preserves_template_imports_for_function_only_output(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5

    checkpoint_dir = tmp_path / "checkpoints"
    result = TraceAADV5(
        llm=FunctionOnlyNumpyLLM(),
        evaluation=ExecutingEvaluation(),
        max_sample_nums=1,
        n_init=1,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text())
    assert result.n_valid_nodes == 1
    assert payload["graph"]["nodes"][0]["code"].startswith("import numpy as np")


def test_traceaad_v5_accepts_small_top_level_helper_functions(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5

    checkpoint_dir = tmp_path / "checkpoints"
    result = TraceAADV5(
        llm=HelperFunctionLLM(),
        evaluation=ExecutingEvaluation(),
        max_sample_nums=1,
        n_init=1,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text())
    assert result.best_node is not None
    assert result.best_node.fitness == 5.0
    assert "def offset" in payload["graph"]["nodes"][0]["code"]


def test_traceaad_v5_logs_structured_evaluation_failures(tmp_path: Path) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADProfiler, TraceAADV5

    log_dir = tmp_path / "logs"
    TraceAADV5(
        llm=RuntimeFailureLLM(),
        evaluation=ExecutingEvaluation(),
        profiler=TraceAADProfiler(
            log_dir=str(log_dir), log_style="simple", create_random_path=False
        ),
        max_sample_nums=1,
        n_init=1,
        checkpoint_dir=log_dir / "checkpoints",
    ).run()

    events = [
        json.loads(line)
        for line in (log_dir / "method_events.jsonl").read_text().splitlines()
    ]
    failure = next(event for event in events if event["event"] == "program_evaluated")
    assert failure["status"] == "eval_failed"
    assert failure["failure_kind"] == "runtime_error"
    assert failure["error_type"] == "ValueError"
    assert failure["error"] == "generated failure"


def test_traceaad_v5_never_lets_parsimony_override_a_better_long_program(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    method = TraceAADV5(
        llm=LongProgramLLM(),
        evaluation=MarkerEvaluation(),
        max_sample_nums=2,
        n_init=1,
        actions_per_iteration=1,
        max_active_trajectories=1,
        operators=(TraceIdeateOp,),
        random_seed=0,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    result = method.run()

    assert result.best_node is not None
    assert result.best_node.fitness == 2.0
    assert result.best_node.program_loc > 100
    assert any(
        result.best_node.id in route.node_ids for route in method.active_trajectories()
    )


def test_traceaad_v5_each_operator_applies_its_direction_to_every_action(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import (
        TraceIdeateOp,
        TraceRefineOp,
        TraceSynthesizeOp,
        TraceTransferOp,
    )

    for operator in (
        TraceIdeateOp,
        TraceRefineOp,
        TraceSynthesizeOp,
        TraceTransferOp,
    ):
        llm = V5MechanismLLM()
        needs_reference = operator in (TraceSynthesizeOp, TraceTransferOp)
        n_init = 2 if needs_reference else 1
        TraceAADV5(
            llm=llm,
            evaluation=IncreasingEvaluation(),
            max_sample_nums=n_init + 1,
            n_init=n_init,
            actions_per_iteration=2,
            max_active_trajectories=4,
            operators=(operator,),
            random_seed=2,
            checkpoint_dir=tmp_path / operator.__name__,
        ).run()

        action_prompt = next(
            prompt for prompt in llm.prompts if "[Action Contract]" in prompt
        )
        direction = action_prompt.split("[Improvement Direction]\n", 1)[1].split(
            "\n\n", 1
        )[0]
        assert direction.startswith("For each action,")

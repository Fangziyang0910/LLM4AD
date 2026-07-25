"""Public mechanism tests for the current TraceAAD V5.1 contract."""

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
        if "[Recent Code Changes]" in prompt:
            return (
                "- A deterministic offset has coincided with improvement.\n"
                "- Continue testing nearby offsets."
            )
        if "Generate a complete implementation" in prompt:
            return self._program(self.calls)
        if "[Requested Modification]" in prompt:
            return self._program(self.calls)
        return "1. Adapt one deterministic offset without adding branches."


class PlainReflectionLLM(ScriptedV5LLM):
    REFLECTION = (
        "- Small deterministic adjustments have improved the current algorithm.\n"
        "- Continue exploring nearby changes while avoiding unnecessary branches."
    )

    def draw_sample(self, prompt, *args, **kwargs):
        self.calls += 1
        self.prompts.append(str(prompt))
        if "[Recent Code Changes]" in prompt:
            return self.REFLECTION
        if "Generate a complete implementation" in prompt:
            return self._program(self.calls)
        if "[Requested Modification]" in prompt:
            return self._program(self.calls)
        return "1. Adapt one deterministic offset without adding branches."


class RecoveringReflectionLLM(V5MechanismLLM):
    def __init__(self) -> None:
        super().__init__()
        self.reflection_calls = 0

    def draw_sample(self, prompt, *args, **kwargs):
        if "[Recent Code Changes]" in prompt:
            self.calls += 1
            self.prompts.append(str(prompt))
            self.reflection_calls += 1
            if self.reflection_calls == 1:
                raise RuntimeError("temporary reflection failure")
            return "- The second stage supports revisiting offsets."
        return super().draw_sample(prompt, *args, **kwargs)


class TwoActionReflectionLLM(V5MechanismLLM):
    def draw_sample(self, prompt, *args, **kwargs):
        if "[Action Contract]" in prompt:
            self.calls += 1
            self.prompts.append(str(prompt))
            long_change = "尝试调整当前算法状态中的一个局部决策规则，观察真实适应度反馈。" * 8
            return f"1. {long_change}\n2. {long_change}"
        return super().draw_sample(prompt, *args, **kwargs)


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
            body = "".join(
                f"    workspace_{index} = value\n" for index in range(120)
            )
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


class SingleActionLLM(ScriptedV5LLM):
    def draw_sample(self, prompt, *args, **kwargs):
        if "[Action Contract]" in prompt:
            return "1. Try one explicit hypothesis even when only one action is returned."
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
    assert payload["format_version"] == 9
    assert all(
        {"program_loc", "code_hash"} <= set(node) for node in payload["graph"]["nodes"]
    )
    assert all(
        len(route["node_ids"]) == len(route["edge_ids"]) + 1
        for route in payload["memory"]["trajectories"]
    )
    assert payload["graph"]["edges"][0]["action"].startswith("Add one")


def test_traceaad_v5_keeps_full_paths_beyond_the_prompt_window(
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
        max_trajectory_length=1,
        max_active_trajectories=1,
        operators=(TraceIdeateOp,),
        global_reflection_code_batch=40,
        random_seed=3,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    longest = max(
        payload["memory"]["trajectories"],
        key=lambda route: len(route["node_ids"]),
    )
    assert len(longest["node_ids"]) == 5
    assert len(longest["edge_ids"]) == 4


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
        global_reflection_code_batch=40,
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
    assert any(
        "[Reference Program]" in prompt
        for prompt in llm.prompts
    )


def test_traceaad_v5_reflects_after_a_fixed_number_of_evaluated_codes(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    llm = PlainReflectionLLM()
    checkpoint_dir = tmp_path / "checkpoints"
    result = TraceAADV5(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=4,
        n_init=1,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceIdeateOp,),
        global_reflection_code_batch=2,
        random_seed=4,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    reflection_prompt = next(
        prompt for prompt in llm.prompts if "[Recent Code Changes]" in prompt
    )
    action_prompts = [
        prompt for prompt in llm.prompts if "[Action Contract]" in prompt
    ]

    assert result.n_experience_updates == 1
    assert reflection_prompt.count("\nChange ") == 2
    assert "[Action]" in reflection_prompt
    assert "[Implementation Idea]" in reflection_prompt
    assert "[Fitness Result]" in reflection_prompt
    assert payload["global_experience"] == PlainReflectionLLM.REFLECTION
    assert PlainReflectionLLM.REFLECTION not in action_prompts[1]
    assert PlainReflectionLLM.REFLECTION in action_prompts[2]


def test_traceaad_v5_reflection_failure_does_not_disable_the_next_stage(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    llm = RecoveringReflectionLLM()
    checkpoint_dir = tmp_path / "checkpoints"
    result = TraceAADV5(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=23,
        n_init=1,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceIdeateOp,),
        global_reflection_code_batch=20,
        random_seed=9,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    reflection_prompts = [
        prompt for prompt in llm.prompts if "[Recent Code Changes]" in prompt
    ]
    action_prompts = [
        prompt for prompt in llm.prompts if "[Action Contract]" in prompt
    ]
    assert llm.reflection_calls == 2
    assert result.n_experience_updates == 1
    assert payload["experience_reflection_attempts"] == 2
    assert "The second stage supports revisiting offsets." in action_prompts[21]
    assert all("def choose" not in prompt for prompt in reflection_prompts)


def test_traceaad_v5_forty_code_reflection_fits_the_real_32k_context(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    llm = TwoActionReflectionLLM()
    result = TraceAADV5(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=43,
        n_init=2,
        actions_per_iteration=2,
        max_active_trajectories=4,
        operators=(TraceIdeateOp,),
        global_reflection_code_batch=40,
        max_context_tokens=32768,
        global_reflection_max_tokens=1024,
        random_seed=11,
        checkpoint_dir=tmp_path / "checkpoints",
    ).run()

    reflection_prompt = next(
        prompt for prompt in llm.prompts if "[Recent Code Changes]" in prompt
    )
    assert result.n_experience_updates == 1
    assert reflection_prompt.count("\nChange ") == 40
    assert reflection_prompt.count("\n[Fitness Result]") == 40
    assert len(reflection_prompt.encode("utf-8")) <= 32768 - 1024


def test_traceaad_v5_checkpoint_resumes_search_state(tmp_path: Path) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    checkpoint_dir = tmp_path / "checkpoints"
    first = TraceAADV5(
        llm=V5MechanismLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=3,
        n_init=2,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceIdeateOp,),
        global_reflection_code_batch=40,
        random_seed=7,
        checkpoint_dir=checkpoint_dir,
    ).run()
    resumed = TraceAADV5(
        llm=V5MechanismLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=5,
        n_init=2,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceIdeateOp,),
        global_reflection_code_batch=40,
        random_seed=999,
        resume_from=checkpoint_dir,
    ).run()

    assert first.n_samples == 3
    assert resumed.n_samples == 5
    assert resumed.n_total_nodes == first.n_total_nodes + 2


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
        global_reflection_code_batch=40,
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
        global_reflection_code_batch=40,
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
        global_reflection_code_batch=40,
        max_stalled_iterations=1,
        random_seed=0,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    action_prompts = [
        prompt for prompt in llm.prompts if "[Action Contract]" in prompt
    ]
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
        global_reflection_code_batch=40,
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
    assert "Do not assume hidden state or change the function signature" in action_prompt


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
        global_reflection_code_batch=40,
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
        global_reflection_code_batch=40,
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
    assert all(
        internal not in action_prompt
        for internal in ("Node p", "anchor_role=", "knowledge provenance", "operator=")
    )
    assert "[Current Program]" in code_prompt
    assert all(
        irrelevant not in code_prompt
        for irrelevant in (
            "Trajectory",
            "Reference",
            "Operator",
            "Node p",
            "anchor_role",
            "provenance",
        )
    )


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
        global_reflection_code_batch=40,
        random_seed=3,
        checkpoint_dir=tmp_path / "checkpoints",
    ).run()

    action_prompts = [
        prompt for prompt in llm.prompts if "[Action Contract]" in prompt
    ]

    assert any("Planned:" in prompt for prompt in action_prompts[1:])
    assert any("Implemented:" in prompt for prompt in action_prompts[1:])
    assert any("Code change:" in prompt for prompt in action_prompts[1:])


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
        global_reflection_code_batch=40,
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
    assert all(
        "no more than 300 characters" in prompt for prompt in generation_prompts
    )


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
        global_reflection_code_batch=40,
        random_seed=0,
        checkpoint_dir=tmp_path / "checkpoints",
    ).run()

    assert result.n_samples == 3
    assert result.n_valid_nodes == 2
    assert result.n_trajectories == 2
    assert result.n_edges == 0


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
        global_reflection_code_batch=40,
        random_seed=0,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    result = method.run()

    assert result.best_node is not None
    assert result.best_node.fitness == 2.0
    assert result.best_node.program_loc > 100
    assert any(
        result.best_node.id in route.node_ids
        for route in method.active_trajectories()
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
            global_reflection_code_batch=40,
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

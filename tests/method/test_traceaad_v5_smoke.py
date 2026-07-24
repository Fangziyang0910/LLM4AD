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
        return json.dumps(
            [
                {
                    "relation": "continue",
                    "evidence_edges": [],
                    "reference_evidence_edges": [],
                    "change": "Add one deterministic offset to the current rule.",
                    "novel_difference": "",
                }
            ]
        )

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
        if "[Old Global Experience]" in prompt:
            return json.dumps(
                [
                    {
                        "kind": "explore",
                        "statement": "A deterministic offset has coincided with improvement.",
                        "condition": "under the current toy evaluator",
                        "evidence_edges": ["e0"],
                    }
                ]
            )
        if "Generate a complete implementation" in prompt:
            return self._program(self.calls)
        if "[Requested Modification]" in prompt:
            return self._program(self.calls)
        relation = "transfer" if "name=trace_transfer" in prompt else "continue"
        return json.dumps(
            [
                {
                    "relation": relation,
                    "evidence_edges": [],
                    "reference_evidence_edges": [],
                    "change": "Adapt one deterministic offset without adding branches.",
                    "novel_difference": "",
                }
            ]
        )


class ParsimonyLLM(LLM):
    def __init__(self) -> None:
        super().__init__()
        self.program_draws = 0

    def draw_sample(self, prompt, *args, **kwargs):
        if "[Action Contract]" in prompt:
            return json.dumps(
                [
                    {
                        "relation": "consolidate",
                        "evidence_edges": [],
                        "reference_evidence_edges": [],
                        "change": "Remove redundant assignments.",
                        "novel_difference": "",
                    }
                ]
            )
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


class PartiallyValidActionLLM(ScriptedV5LLM):
    def draw_sample(self, prompt, *args, **kwargs):
        if "[Action Contract]" in prompt:
            return json.dumps(
                [
                    {
                        "relation": "continue",
                        "evidence_edges": ["e999"],
                        "reference_evidence_edges": [],
                        "change": "This object has a fabricated citation.",
                        "novel_difference": "",
                    },
                    {
                        "relation": "redirect",
                        "evidence_edges": [],
                        "reference_evidence_edges": [],
                        "change": "Try one uncited but explicit hypothesis.",
                        "novel_difference": "",
                    },
                ]
            )
        return super().draw_sample(prompt, *args, **kwargs)


def test_traceaad_v5_has_an_independent_public_method_class() -> None:
    from llm4ad.method.traceaad import TraceAAD
    from llm4ad.method.traceaad_v5 import TraceAADV5

    assert TraceAADV5 is not TraceAAD
    assert TraceAADV5.__module__.startswith("llm4ad.method.traceaad_v5")


def test_traceaad_v5_runs_structured_actions_and_writes_v5_state(
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
    assert payload["format_version"] == 6
    assert all(
        {"program_loc", "code_hash"} <= set(node) for node in payload["graph"]["nodes"]
    )
    assert all(
        len(route["node_ids"]) == len(route["edge_ids"]) + 1
        for route in payload["memory"]["trajectories"]
    )
    assert payload["graph"]["edges"][0]["change"].startswith("Add one")


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
        experience_batch_size=20,
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
        experience_batch_size=20,
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
        "[Reference Program: knowledge only, never a parent]" in prompt
        for prompt in llm.prompts
    )


def test_traceaad_v5_global_experience_updates_and_guides_later_actions(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    llm = V5MechanismLLM()
    checkpoint_dir = tmp_path / "checkpoints"
    result = TraceAADV5(
        llm=llm,
        evaluation=IncreasingEvaluation(),
        max_sample_nums=5,
        n_init=2,
        actions_per_iteration=1,
        max_active_trajectories=4,
        operators=(TraceIdeateOp,),
        experience_batch_size=2,
        random_seed=4,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    assert result.n_experience_updates == 1
    assert payload["global_experience_entries"][0]["evidence_edge_ids"] == [0]
    assert any(
        "A deterministic offset has coincided with improvement." in prompt
        and "[Action Contract]" in prompt
        for prompt in llm.prompts
    )


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
        experience_batch_size=20,
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
        experience_batch_size=20,
        random_seed=999,
        resume_from=checkpoint_dir,
    ).run()

    assert first.n_samples == 3
    assert resumed.n_samples == 5
    assert resumed.n_total_nodes == first.n_total_nodes + 2


def test_traceaad_v5_prefers_shorter_exact_ties_without_clone_churn(
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
        experience_batch_size=20,
        random_seed=0,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    assert result.best_node is not None
    assert result.best_node.id == 1
    assert result.best_node.program_loc == 2
    assert payload["graph"]["edges"][0]["global_best_update_reason"] == "tie_shorter"
    assert payload["graph"]["edges"][1]["new_global_best"] is False


def test_traceaad_v5_executes_the_valid_subset_of_structured_actions(
    tmp_path: Path,
) -> None:
    from llm4ad.method.traceaad_v5 import TraceAADV5
    from llm4ad.method.traceaad_v5.operators import TraceIdeateOp

    checkpoint_dir = tmp_path / "checkpoints"
    result = TraceAADV5(
        llm=PartiallyValidActionLLM(),
        evaluation=IncreasingEvaluation(),
        max_sample_nums=2,
        n_init=1,
        actions_per_iteration=2,
        max_active_trajectories=2,
        operators=(TraceIdeateOp,),
        experience_batch_size=20,
        random_seed=0,
        checkpoint_dir=checkpoint_dir,
    ).run()

    payload = json.loads((checkpoint_dir / "latest.json").read_text(encoding="utf-8"))
    assert result.n_samples == 2
    assert result.n_edges == 1
    assert payload["graph"]["edges"][0]["change"] == (
        "Try one uncited but explicit hypothesis."
    )

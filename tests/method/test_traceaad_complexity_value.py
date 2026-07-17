"""Tests for TraceAAD complexity metrics and C/R value dimensions."""
from __future__ import annotations

import pickle
import unittest

from llm4ad.method.traceaad.complexity import analyze_code_complexity
from llm4ad.method.traceaad.context import _experience_block, build_action_prompt
from llm4ad.method.traceaad.derivation_graph import DerivationGraph
from llm4ad.method.traceaad.experience_memory import ExperienceMemory
from llm4ad.method.traceaad.schema import EvalResult, ExperienceBatch, ExperienceExample, ValueVec
from llm4ad.method.traceaad.trajectory_memory import TrajectoryMemory
from llm4ad.method.traceaad.value import (
    ValueWeights,
    compute_value_vec,
    robust_active_metric_bounds,
    scalarize,
)


SIMPLE = """
def select_next_node(current, destination, unvisited):
    best = None
    best_score = None
    for node in unvisited:
        score = destination[current][node]
        if best is None or score < best_score:
            best = node
            best_score = score
    return best
"""

COMPLEX = """
def select_next_node(current, destination, unvisited):
    best = None
    best_score = None
    for node in unvisited:
        score = 0.0
        for other in unvisited:
            if other == node:
                continue
            local = destination[current][node]
            if local < 0.1:
                score += local * 2.0
            elif local < 0.5:
                score += local
            else:
                score += local * 0.5
            if other % 2 == 0:
                score -= 0.01
            else:
                score += 0.01
        if best is None or score < best_score:
            best = node
            best_score = score
    return best
"""


class ComplexityAndValueTests(unittest.TestCase):
    def test_analyze_code_complexity_returns_composite_score(self):
        metrics = analyze_code_complexity(SIMPLE)
        self.assertIn("complexity_score", metrics)
        self.assertIn("cyclomatic_complexity", metrics)
        self.assertGreater(metrics["complexity_score"], 0.0)
        self.assertLessEqual(metrics["complexity_score"], 1.0)

    def test_more_branching_code_has_higher_or_equal_complexity(self):
        simple = analyze_code_complexity(SIMPLE)["complexity_score"]
        complex_score = analyze_code_complexity(COMPLEX)["complexity_score"]
        self.assertGreaterEqual(complex_score, simple)

    def test_eval_result_is_process_serializable(self):
        result = EvalResult(
            fitness=1.0,
            complexity=0.2,
            runtime=0.1,
            complexity_metrics={"lines_of_code": 5.0},
        )

        restored = pickle.loads(pickle.dumps(result))

        self.assertEqual(restored, result)

    def test_value_vec_includes_compactness_and_speed(self):
        graph = DerivationGraph()
        memory = TrajectoryMemory(max_trajectory_length=8)
        a = graph.add_node(
            code=SIMPLE, idea="simple", fitness=1.0,
            complexity=0.1, runtime=0.05, complexity_metrics={"cyclomatic_complexity": 2.0},
        )
        b = graph.add_node(
            code=COMPLEX, idea="complex", fitness=1.0,
            complexity=0.8, runtime=0.50, complexity_metrics={"cyclomatic_complexity": 8.0},
        )
        ta = memory.create_initial(node_id=a.id, island_id=0)
        tb = memory.create_initial(node_id=b.id, island_id=1)
        actives = memory.active()
        cmin, cmax = robust_active_metric_bounds(
            trajectories=actives, graph=graph, kind="complexity",
        )
        rmin, rmax = robust_active_metric_bounds(
            trajectories=actives, graph=graph, kind="runtime",
        )
        va = compute_value_vec(
            trajectory=ta, graph=graph, active_others=(tb,),
            fmin=1.0, fmax=1.0, maximize=True, w=ValueWeights(),
            cmin=cmin, cmax=cmax, rmin=rmin, rmax=rmax,
        )
        vb = compute_value_vec(
            trajectory=tb, graph=graph, active_others=(ta,),
            fmin=1.0, fmax=1.0, maximize=True, w=ValueWeights(),
            cmin=cmin, cmax=cmax, rmin=rmin, rmax=rmax,
        )
        self.assertEqual(len(va.as_tuple()), 6)
        self.assertGreater(va.compactness, vb.compactness)
        self.assertGreater(va.speed, vb.speed)
        self.assertGreater(scalarize(va, ValueWeights()), scalarize(vb, ValueWeights()))

    def test_causal_narrative_includes_structure_and_runtime_deltas(self):
        from llm4ad.base import Function

        graph = DerivationGraph()
        memory = TrajectoryMemory(max_trajectory_length=8)
        parent = graph.add_node(
            code=SIMPLE, idea="parent", fitness=1.0,
            complexity=0.2, runtime=0.10,
            complexity_metrics={"cyclomatic_complexity": 2.0, "max_nesting_depth": 1.0, "lines_of_code": 10.0},
        )
        child = graph.add_node(
            code=COMPLEX, idea="child", fitness=1.1,
            complexity=0.6, runtime=0.20,
            complexity_metrics={"cyclomatic_complexity": 6.0, "max_nesting_depth": 3.0, "lines_of_code": 20.0},
        )
        edge = graph.add_edge(
            parent_id=parent.id, child_id=child.id, action="add density score",
            operator="endpoint_refine",
            delta=0.1, outcome="improve",
        )
        traj = memory.create_initial(node_id=parent.id, island_id=0)
        traj = memory.extend(
            trajectory_id=traj.id,
            parent_id=parent.id,
            child_id=child.id,
            edge_id=edge.id,
        )
        prompt = build_action_prompt(
            graph=graph,
            trajectory=traj,
            base_node_id=child.id,
            base_reason="endpoint",
            operator_name="endpoint_refine",
            operator_role="exploit",
            operator_constraint="refine",
            experience_memory=ExperienceMemory(graph),
            contrast=None,
            task_description="TSP constructive heuristic.",
            template_function=Function(
                name="select_next_node",
                args="current, destination, unvisited",
                body="    pass\n",
            ),
            action_count=2,
            maximize=True,
        )
        self.assertIn("structure:", prompt)
        self.assertIn("runtime:", prompt)
        self.assertIn("ΔC=", prompt)
        self.assertIn("ΔR=", prompt)
        self.assertIn("Structure/runtime:", prompt)
        self.assertIn("[Past Action Evidence]", prompt)
        self.assertNotIn("mech=", prompt)
        self.assertNotIn("Mechanism Patterns", prompt)

    def test_pareto_tuple_length_is_six(self):
        value = ValueVec(
            quality=1.0, potential=0.5, diversity=0.5, novelty=0.5,
            compactness=0.8, speed=0.7,
        )
        self.assertEqual(value.as_tuple(), (1.0, 0.5, 0.5, 0.5, 0.8, 0.7))

    def test_experience_prompt_limits_count_and_action_length(self):
        def example(index: int, outcome: str) -> ExperienceExample:
            return ExperienceExample(
                edge_id=index,
                operator="endpoint_refine",
                action=f"action-{index}-" + "x" * 400,
                delta=1.0 if outcome == "improve" else -1.0,
                outcome=outcome,
                iteration=index,
            )

        batch = ExperienceBatch(
            positives=tuple(example(i, "improve") for i in range(2)),
            negatives=tuple(example(i + 2, "regress") for i in range(2)),
        )

        block = _experience_block(batch, max_action_chars=300)

        action_lines = [line for line in block.splitlines() if line.startswith("- [operator=")]
        self.assertEqual(len(action_lines), 4)
        for line in action_lines:
            action = line.split(" action=", 1)[1].rsplit(" delta=", 1)[0]
            self.assertEqual(len(action), 300)
            self.assertTrue(action.endswith("..."))

    def test_empty_experience_prompt_has_short_empty_state(self):
        self.assertEqual(
            _experience_block(ExperienceBatch()),
            "No successful or failed past actions recorded yet.",
        )


if __name__ == "__main__":
    unittest.main()

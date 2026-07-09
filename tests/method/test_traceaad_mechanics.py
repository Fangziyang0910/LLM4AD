import unittest

from llm4ad.base import Evaluation, LLM
from llm4ad.method.traceaad import TraceAAD
from llm4ad.method.traceaad.derivation_graph import DerivationGraph
from llm4ad.method.traceaad.prompt import TraceAADPrompt
from llm4ad.method.traceaad.trajectory_branching import select_base_node
from llm4ad.method.traceaad.trajectory_library import TrajectoryLibrary
from llm4ad.method.traceaad.trajectory_scorer import compute_trajectory_score


TEMPLATE = """
def heuristic(x):
    return 0
"""


class CountingEvaluation(Evaluation):
    def __init__(self):
        super().__init__(
            template_program=TEMPLATE,
            task_description="Design a heuristic(x) function. Higher score is better.",
            safe_evaluate=False,
        )
        self.calls = 0

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs):
        self.calls += 1
        if callable_func is None:
            return None
        return float(callable_func(0))


class ScriptedLLM(LLM):
    def __init__(self):
        super().__init__(do_auto_trim=False)
        self.prompts = []
        self.closed = False

    def draw_sample(self, prompt: str, *args, **kwargs) -> str:
        self.prompts.append(prompt)
        if "Generate a complete implementation" in prompt:
            return _program_response("Initial constant heuristic", 1)
        if "next-step modifications" in prompt:
            return "1. Increase the returned score with a simple constant"
        if "Implement the requested modification" in prompt:
            return _program_response("Larger constant heuristic", 2)
        raise AssertionError(f"unexpected prompt: {prompt[:100]}")

    def close(self):
        self.closed = True


def _program_response(idea: str, value: int) -> str:
    return f"""Idea: {idea}
Code:
```python
def heuristic(x):
    return {value}
```
"""


def _node(graph: DerivationGraph, fitness: float | None):
    return graph.add_node(
        code=f"def heuristic(x): return {fitness}",
        idea=f"fitness {fitness}",
        fitness=fitness,
        is_valid=fitness is not None,
    )


def _trajectory(graph: DerivationGraph, fitnesses: list[float | None]):
    library = TrajectoryLibrary(max_trajectory_length=8)
    nodes = [_node(graph, fitness) for fitness in fitnesses]
    trajectory = library.create_initial(node_id=nodes[0].id)
    edges = []
    for parent, child in zip(nodes, nodes[1:]):
        edge = graph.add_edge(
            parent_id=parent.id,
            child_id=child.id,
            action=f"p{parent.id} to p{child.id}",
        )
        edges.append(edge)
        trajectory = library.extend(trajectory_id=trajectory.id, edge=edge)
    return library, trajectory, nodes, edges


class TraceAADMechanicsTest(unittest.TestCase):
    def test_traceaad_runs_one_iteration_through_llm4ad_interfaces(self):
        llm = ScriptedLLM()
        evaluation = CountingEvaluation()
        method = TraceAAD(
            llm=llm,
            evaluation=evaluation,
            max_sample_nums=2,
            n_init=1,
            n_iterations=1,
            actions_per_iteration=1,
            top_k=1,
            num_evaluators=1,
        )

        result = method.run()

        self.assertTrue(llm.closed)
        self.assertEqual(evaluation.calls, 2)
        self.assertEqual(result.n_samples, 2)
        self.assertEqual(result.n_total_nodes, 2)
        self.assertEqual(result.n_valid_nodes, 2)
        self.assertEqual(result.n_edges, 1)
        self.assertEqual(result.n_trajectories, 2)
        self.assertEqual(result.best_node.fitness, 2.0)
        self.assertEqual([node.fitness for node in method.graph.nodes()], [1.0, 2.0])
        self.assertIn("[Base Program To Modify]", llm.prompts[1])
        self.assertIn("Selection reason: initial", llm.prompts[1])
        self.assertIn("[Target Function Contract]", llm.prompts[2])

    def test_parse_program_response_accepts_boxed_idea_and_function_code(self):
        method = TraceAAD(
            llm=ScriptedLLM(),
            evaluation=CountingEvaluation(),
            max_sample_nums=1,
            n_init=0,
            n_iterations=0,
        )
        response = """\\boxed{Use a constant heuristic}
```python
def heuristic(x):
    return 3
```
"""

        generated = method._parse_program_response(response)

        self.assertIsNotNone(generated)
        self.assertEqual(generated.idea, "Use a constant heuristic")
        self.assertIn("return 3", str(generated.program))

    def test_select_base_node_branches_after_regression(self):
        graph = DerivationGraph()
        library, trajectory, nodes, edges = _trajectory(graph, [1.0, 3.0, 2.0])
        selection = select_base_node(graph=graph, trajectory=trajectory, maximize=True)
        child = _node(graph, 4.0)
        branch_edge = graph.add_edge(parent_id=selection.node_id, child_id=child.id, action="try branch")

        branch = library.branch_from(
            trajectory_id=trajectory.id,
            base_node_id=selection.node_id,
            edge=branch_edge,
        )

        self.assertEqual(selection.node_id, nodes[1].id)
        self.assertEqual(selection.reason, "last_regressed")
        self.assertEqual(branch.node_ids, (nodes[0].id, nodes[1].id, child.id))
        self.assertEqual(branch.edge_ids, (edges[0].id, branch_edge.id))

    def test_stepwise_score_prefers_consistent_path_with_same_endpoint(self):
        graph = DerivationGraph()
        library = TrajectoryLibrary()

        jump_start = graph.add_node(code="a0", idea="a0", fitness=2.0, is_valid=True)
        jump_mid = graph.add_node(code="a1", idea="a1", fitness=8.0, is_valid=True)
        jump_end = graph.add_node(code="a2", idea="a2", fitness=7.0, is_valid=True)
        jump_traj = library.create_initial(node_id=jump_start.id)
        jump_edge_1 = graph.add_edge(parent_id=jump_start.id, child_id=jump_mid.id, action="large gain")
        jump_traj = library.extend(trajectory_id=jump_traj.id, edge=jump_edge_1)
        jump_edge_2 = graph.add_edge(parent_id=jump_mid.id, child_id=jump_end.id, action="regression")
        jump_traj = library.extend(trajectory_id=jump_traj.id, edge=jump_edge_2)

        steady_start = graph.add_node(code="b0", idea="b0", fitness=2.0, is_valid=True)
        steady_mid = graph.add_node(code="b1", idea="b1", fitness=4.5, is_valid=True)
        steady_end = graph.add_node(code="b2", idea="b2", fitness=7.0, is_valid=True)
        steady_traj = library.create_initial(node_id=steady_start.id)
        steady_edge_1 = graph.add_edge(parent_id=steady_start.id, child_id=steady_mid.id, action="steady gain")
        steady_traj = library.extend(trajectory_id=steady_traj.id, edge=steady_edge_1)
        steady_edge_2 = graph.add_edge(parent_id=steady_mid.id, child_id=steady_end.id, action="steady gain")
        steady_traj = library.extend(trajectory_id=steady_traj.id, edge=steady_edge_2)

        jump_score = compute_trajectory_score(
            trajectory=jump_traj,
            graph=graph,
            total_visits=0,
            iteration=0,
            max_iterations=10,
            fitness_min=0.0,
            fitness_max=10.0,
            c0=0.0,
            maximize=True,
        )
        steady_score = compute_trajectory_score(
            trajectory=steady_traj,
            graph=graph,
            total_visits=0,
            iteration=0,
            max_iterations=10,
            fitness_min=0.0,
            fitness_max=10.0,
            c0=0.0,
            maximize=True,
        )

        self.assertEqual(graph.get_node(jump_traj.endpoint_id).fitness, graph.get_node(steady_traj.endpoint_id).fitness)
        self.assertGreater(steady_score, jump_score)

    def test_action_prompt_uses_selected_base_node_program(self):
        graph = DerivationGraph()
        library, trajectory, nodes, _edges = _trajectory(graph, [1.0, 3.0, 2.0])
        prompt = TraceAADPrompt.build_action_prompt(
            graph=graph,
            trajectory=trajectory,
            task_description="Improve heuristic.",
            maximize=True,
            base_node_id=nodes[1].id,
            base_selection_reason="last_regressed",
            action_count=2,
        )

        self.assertIn("[Base Program To Modify]", prompt)
        self.assertIn("Selection reason: last_regressed", prompt)
        self.assertIn("def heuristic(x): return 3.0", prompt)
        self.assertNotIn("def heuristic(x): return 2.0", prompt.split("[Base Program To Modify]", 1)[1])


if __name__ == "__main__":
    unittest.main()

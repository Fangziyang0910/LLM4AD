import unittest

from llm4ad.base import Function, TextFunctionProgramConverter
from llm4ad.method.mcts_ahd.mcts import MCTS, MCTSNode
from llm4ad.method.mcts_ahd.population import Population
from llm4ad.method.mcts_trace_ahd import MCTS_Trace_AHD
from llm4ad.method.mcts_trace_ahd.prompt import MCTSTracePrompt


def make_function(label: int, score=None) -> Function:
    func = Function(name="heuristic", args="x", body=f"    return {label}")
    func.algorithm = f"algorithm-{label}"
    func.score = score
    return func


def make_method() -> MCTS_Trace_AHD:
    method = object.__new__(MCTS_Trace_AHD)
    method._task_description_str = "Design a heuristic."
    method._function_to_evolve = make_function(0)
    method._template_program = TextFunctionProgramConverter.text_to_program(str(method._function_to_evolve))
    method._population = Population(init_pop_size=4, pop_size=10)
    method._pop_size = 10
    method._e1_min_refs = MCTS_Trace_AHD.E1_MIN_REFS
    method._e1_max_refs = MCTS_Trace_AHD.E1_MAX_REFS
    method._max_sample_nums = 1000
    method._tot_sample_nums = 0
    method._consecutive_sample_failures = 0
    method._max_consecutive_sample_failures = 20
    method._search_aborted = False
    method._debug_mode = False
    method._profiler = None
    return method


def attach_node(parent: MCTSNode, func: Function, depth: int) -> MCTSNode:
    node = MCTSNode(
        func.algorithm,
        str(func),
        -1 * func.score,
        individual=func,
        parent=parent,
        depth=depth,
        visit=1,
        Q=func.score,
        raw_info=func,
    )
    parent.add_child(node)
    return node


class MCTSTraceAHDMechanicsTest(unittest.TestCase):
    def test_trace_states_are_ordered_and_compare_to_previous_state(self):
        states = MCTSTracePrompt.build_trace_states(
            [make_function(1, 1.0), make_function(2, 3.0), make_function(3, 2.0)]
        )

        self.assertEqual([state.description for state in states], ["algorithm-1", "algorithm-2", "algorithm-3"])
        self.assertEqual([state.score for state in states], [1.0, 3.0, 2.0])
        self.assertEqual([state.change for state in states], ["start", "improved", "regressed"])

    def test_ordered_trace_from_node_preserves_parent_child_order(self):
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        n1 = attach_node(mcts.root, make_function(1, 1.0), depth=1)
        n2 = attach_node(n1, make_function(2, 2.0), depth=2)
        n3 = attach_node(n2, make_function(3, 3.0), depth=3)

        trace = method._ordered_trace_from_node(n3)

        self.assertEqual([func.algorithm for func in trace], ["algorithm-1", "algorithm-2", "algorithm-3"])

    def test_s1_prompt_uses_trace_without_historical_code_dump(self):
        prompt = MCTSTracePrompt.get_prompt_s1_trace(
            "Task.",
            [make_function(1, 1.0), make_function(2, 2.0), make_function(3, 1.5)],
            make_function(3, 1.5),
            make_function(0),
        )

        self.assertIn("[Algorithm Improvement Trace]", prompt)
        self.assertIn("State 1:", prompt)
        self.assertIn("Change from previous: start", prompt)
        self.assertIn("Change from previous: improved", prompt)
        self.assertIn("Change from previous: regressed", prompt)
        self.assertIn("[Current Algorithm To Improve]", prompt)
        self.assertNotIn("return 1", prompt)
        self.assertNotIn("return 2", prompt)
        self.assertIn("return 3", prompt)

    def test_m1_and_m2_expansion_include_trace_context(self):
        for op in ("m1", "m2"):
            with self.subTest(op=op):
                method = make_method()
                mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
                n1 = attach_node(mcts.root, make_function(1, 1.0), depth=1)
                n2 = attach_node(n1, make_function(2, 2.0), depth=2)
                captured = {}

                def sample(prompt, func_only=False, **kwargs):
                    captured["prompt"] = prompt
                    captured["operator"] = kwargs["operator"]
                    return make_function(9, 9.0)

                method._sample_evaluate_register = sample

                method.expand(mcts, [], n2, op)

                self.assertEqual(captured["operator"], op)
                self.assertIn("[Algorithm Improvement Trace]", captured["prompt"])
                self.assertIn("State 1:", captured["prompt"])
                self.assertIn("State 2:", captured["prompt"])
                self.assertIn("Use the ordered trace above", captured["prompt"])

    def test_e2_prompt_remains_without_trace_context(self):
        method = make_method()
        parent_func = make_function(1, 1.0)
        elite = make_function(99, 99.0)
        method._population._population = [parent_func, elite]
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        parent = attach_node(mcts.root, parent_func, depth=1)
        captured = {}

        def sample(prompt, func_only=False, **kwargs):
            captured["prompt"] = prompt
            captured["operator"] = kwargs["operator"]
            return make_function(2, 2.0)

        method._sample_evaluate_register = sample

        method.expand(mcts, [], parent, "e2")

        self.assertEqual(captured["operator"], "e2")
        self.assertNotIn("[Algorithm Improvement Trace]", captured["prompt"])
        self.assertIn("algorithm-99", captured["prompt"])

    def test_s1_noops_for_single_state_trace(self):
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        parent = attach_node(mcts.root, make_function(1, 1.0), depth=1)

        def fail_if_called(prompt, func_only=False, **kwargs):
            raise AssertionError("s1 should not sample with a single-state trace")

        method._sample_evaluate_register = fail_if_called
        before = len(parent.children)

        method.expand(mcts, [], parent, "s1")

        self.assertEqual(len(parent.children), before)


if __name__ == "__main__":
    unittest.main()

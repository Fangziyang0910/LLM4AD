import math
import random
import unittest

from llm4ad.base import Function, TextFunctionProgramConverter
from llm4ad.method.mcts_ahd.mcts import MCTS, MCTSNode
from llm4ad.method.mcts_ahd.mcts_ahd import MCTS_AHD
from llm4ad.method.mcts_ahd.population import Population


def make_function(label: int, score=None) -> Function:
    func = Function(name="heuristic", args="x", body=f"    return {label}")
    func.algorithm = f"algorithm-{label}"
    func.score = score
    return func


def make_method() -> MCTS_AHD:
    method = object.__new__(MCTS_AHD)
    method._task_description_str = "Design a heuristic."
    method._function_to_evolve = make_function(0)
    method._template_program = TextFunctionProgramConverter.text_to_program(str(method._function_to_evolve))
    method._population = Population(init_pop_size=4, pop_size=10)
    method._pop_size = 10
    method._e1_min_refs = MCTS_AHD.E1_MIN_REFS
    method._e1_max_refs = MCTS_AHD.E1_MAX_REFS
    method._max_sample_nums = 1000
    method._tot_sample_nums = 0
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


class ImmediateResult:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        return ImmediateResult(fn(*args, **kwargs))


class FakeSampler:
    def __init__(self, func: Function):
        self.func = func

    def get_thought_and_function(self, task_description, prompt):
        return "aligned description", self.func


class FakeEvaluator:
    def __init__(self, score):
        self.score = score

    def evaluate_program_record_time(self, program):
        return self.score, 0.01


class MCTSAHDMechanicsTest(unittest.TestCase):
    def test_child_depth_increments_from_parent(self):
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        parent = attach_node(mcts.root, make_function(1, 1.0), depth=3)
        parent.subtree.append(parent)
        method._sample_evaluate_register = lambda prompt, func_only=False: make_function(2, 2.0)

        method.expand(mcts, [], parent, "m1")

        self.assertEqual(parent.children[-1].depth, 4)
        self.assertEqual(mcts.max_depth, 10)

    def test_e1_samples_two_to_five_distinct_root_subtrees(self):
        random.seed(7)
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        for i in range(6):
            node = attach_node(mcts.root, make_function(i, float(i)), depth=1)
            node.subtree.append(node)

        refs = method._sample_e1_references_from_root(mcts)

        self.assertGreaterEqual(len(refs), 2)
        self.assertLessEqual(len(refs), 5)
        self.assertEqual(len({ref.algorithm for ref in refs}), len(refs))

    def test_e1_uses_exactly_two_refs_when_two_root_subtrees_exist(self):
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        for i in range(2):
            node = attach_node(mcts.root, make_function(i, float(i)), depth=1)
            node.subtree.append(node)

        refs = method._sample_e1_references_from_root(mcts)

        self.assertEqual(len(refs), 2)

    def test_progressive_root_e1_noops_with_fewer_than_two_subtrees(self):
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        node = attach_node(mcts.root, make_function(1, 1.0), depth=1)
        node.subtree.append(node)

        def fail_if_called(prompt, func_only=False):
            raise AssertionError("e1 should not sample with fewer than two root subtrees")

        method._sample_evaluate_register = fail_if_called
        before = len(mcts.root.children)

        method.expand(mcts, mcts.root.children, mcts.root, "e1")

        self.assertEqual(len(mcts.root.children), before)

    def test_eval_counter_increments_without_profiler(self):
        method = make_method()
        method._sampler = FakeSampler(make_function(3))
        method._evaluator = FakeEvaluator(score=3.5)
        method._evaluation_executor = ImmediateExecutor()

        func = method._sample_evaluate_register("prompt", func_only=True)

        self.assertEqual(method._tot_sample_nums, 1)
        self.assertEqual(func.score, 3.5)

    def test_duplicate_e1_score_skips_child_and_backprop(self):
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        for i, score in enumerate([1.0, 2.0], start=1):
            node = attach_node(mcts.root, make_function(i, score), depth=1)
            node.subtree.append(node)
        method._sample_evaluate_register = lambda prompt, func_only=False: make_function(9, 1.0)
        before_children = len(mcts.root.children)
        before_visits = mcts.root.visits

        method.expand(mcts, mcts.root.children, mcts.root, "e1")

        self.assertEqual(len(mcts.root.children), before_children)
        self.assertEqual(mcts.root.visits, before_visits)

    def test_progressive_widening_triggers_on_equal_threshold(self):
        method = make_method()
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        node = MCTSNode("parent", "code", 0, depth=1, visit=4, Q=0)
        node.children = [
            MCTSNode("child-1", "code-1", 0, parent=node, visit=1, Q=0),
            MCTSNode("child-2", "code-2", 0, parent=node, visit=1, Q=0),
        ]

        self.assertTrue(method._should_progressively_widen(mcts, node))
        node.children.append(MCTSNode("child-3", "code-3", 0, parent=node, visit=1, Q=0))
        self.assertFalse(method._should_progressively_widen(mcts, node))

    def test_uct_with_equal_q_bounds_does_not_crash(self):
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        child = MCTSNode("child", "code", -1, parent=mcts.root, depth=1, visit=1, Q=1.0)
        mcts.root.visits = 1
        mcts.q_min = 1.0
        mcts.q_max = 1.0

        value = mcts.uct(child, eval_remain=1.0)

        self.assertTrue(math.isfinite(value))

    def test_expansion_schedule_and_tree_preserve_low_score_nodes(self):
        expanded_ops = []
        for op, weight in zip(MCTS_AHD.DEFAULT_OPERATORS, MCTS_AHD.DEFAULT_OPERATOR_WEIGHTS):
            expanded_ops.extend([op] * weight)
        self.assertEqual(expanded_ops, ["e2", "m1", "m1", "m2", "m2", "s1"])

        method = make_method()
        method._population = Population(init_pop_size=4, pop_size=1)
        high = make_function(10, 10.0)
        method._population._population = [high]
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        parent = attach_node(mcts.root, make_function(1, 1.0), depth=1)
        parent.subtree.append(parent)
        low = make_function(0, 0.0)
        method._sample_evaluate_register = lambda prompt, func_only=False: low

        method.expand(mcts, [], parent, "m1")

        self.assertIs(parent.children[-1].individual, low)
        self.assertEqual(parent.children[-1].individual.score, 0.0)
        self.assertEqual([func.score for func in method._population.population], [10.0])

    def test_e2_samples_pending_elite_before_survival(self):
        method = make_method()
        parent_func = make_function(1, 1.0)
        pending_elite = make_function(99, 99.0)
        method._population._population = [parent_func]
        method._population._next_gen_pop = [pending_elite]
        mcts = MCTS("Root", alpha=0.5, lambad0=0.1)
        parent = attach_node(mcts.root, parent_func, depth=1)
        captured = {}

        def sample(prompt, func_only=False):
            captured["prompt"] = prompt
            return make_function(2, 2.0)

        method._sample_evaluate_register = sample

        method.expand(mcts, [], parent, "e2")

        self.assertIn("algorithm-99", captured["prompt"])


if __name__ == "__main__":
    unittest.main()

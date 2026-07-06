import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from llm4ad.base import Evaluation, Function, LLM
from llm4ad.method.hsevo.hsevo import HSEvo
from llm4ad.method.hsevo.prompt import HSEvoPrompt
from llm4ad.method.hsevo.sampler import HSEvoSampler


def make_function(label: int, score=None) -> Function:
    func = Function(name="heuristic", args="x", body=f"    return {label}")
    func.score = score
    func.algorithm = f"algorithm-{label}"
    return func


def make_code(label: int) -> str:
    return f"```python\ndef heuristic_v2(x):\n    return {label}\n```"


class ImmediateResult:
    def __init__(self, value):
        self._value = value

    def result(self):
        return self._value


class ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):
        return ImmediateResult(fn(*args, **kwargs))

    def shutdown(self, *args, **kwargs):
        pass


class ScriptedLLM(LLM):
    def __init__(self, responses=None, temperature=None):
        super().__init__()
        self.responses = list(responses or [])
        self.prompts = []
        self.closed = False
        if temperature is not None:
            self.temperature = temperature

    def draw_sample(self, prompt, *args, **kwargs):
        self.prompts.append((prompt, kwargs))
        if not self.responses:
            return "invalid generation"
        return self.responses.pop(0)

    def close(self):
        self.closed = True


class FakeEvaluation(Evaluation):
    def __init__(self, scores=None):
        super().__init__(
            template_program=(
                "import numpy as np\n"
                "def heuristic(x: np.ndarray) -> int:\n"
                "    \"\"\"Pick the next item from x.\"\"\"\n"
                "    return int(x[0])\n"
            ),
            task_description="Design a heuristic.",
            safe_evaluate=False,
        )
        self.scores = list(scores or [])
        self.programs = []

    def evaluate_program(self, program_str, callable_func, **kwargs):
        self.programs.append(program_str)
        if not self.scores:
            return None
        return self.scores.pop(0)


def make_method(
        *,
        responses=None,
        scores=None,
        pop_size=2,
        init_pop_size=2,
        mutation_rate=0.5,
        max_sample_nums=100,
        hm_size=2,
        max_iter=0,
        hs_attempts_per_generation=0,
        llm_temperature=None,
) -> tuple[HSEvo, ScriptedLLM, FakeEvaluation]:
    llm = ScriptedLLM(responses=responses, temperature=llm_temperature)
    evaluation = FakeEvaluation(scores=scores)
    method = HSEvo(
        llm=llm,
        evaluation=evaluation,
        profiler=None,
        max_sample_nums=max_sample_nums,
        pop_size=pop_size,
        init_pop_size=init_pop_size,
        mutation_rate=mutation_rate,
        hm_size=hm_size,
        max_iter=max_iter,
        hs_attempts_per_generation=hs_attempts_per_generation,
        num_samplers=1,
        num_evaluators=1,
    )
    method._evaluation_executor.shutdown(cancel_futures=True)
    method._evaluation_executor = ImmediateExecutor()
    return method, llm, evaluation


class HSEvoMechanicsTest(unittest.TestCase):
    def test_prompt_signature_is_derived_from_llm4ad_template(self):
        method, _, _ = make_method()
        prompt = HSEvoPrompt("Task.", method._function_to_evolve)

        self.assertEqual(prompt.func_signature(2), "def heuristic_v2(x: np.ndarray) -> int:")
        self.assertIn("Pick the next item from x.", prompt.func_desc)
        self.assertNotIn("unvisited_nodes: set", prompt.func_signature(2))

    def test_seed_and_initialization_use_unified_evaluator_and_operator_tags(self):
        method, llm, evaluation = make_method(
            responses=[make_code(1), make_code(2)],
            scores=[100.0, 1.0, 2.0],
            pop_size=2,
            init_pop_size=2,
            llm_temperature=1.0,
        )

        seed = method._evaluate_seed()
        method._initialize_population()

        self.assertEqual(seed.operator, "seed")
        self.assertEqual([func.operator for func in method._population.population], ["init", "init"])
        self.assertEqual(method._tot_sample_nums, 3)
        self.assertEqual(method._elite_function.score, 100.0)
        self.assertEqual(len(evaluation.programs), 3)
        self.assertTrue(all("import random" in program for program in evaluation.programs))
        self.assertTrue(all(kwargs == {"temperature": 1.3} for _, kwargs in llm.prompts))

    def test_invalid_code_and_score_count_budget_but_do_not_enter_population(self):
        method, _, _ = make_method(
            responses=["not valid python", make_code(2), make_code(3)],
            scores=[None, 3.0],
            pop_size=2,
            init_pop_size=3,
        )

        method._initialize_population()

        self.assertEqual(method._tot_sample_nums, 3)
        self.assertEqual([func.score for func in method._population.population], [3.0])
        with self.assertRaisesRegex(RuntimeError, "fewer than two valid functions"):
            method._select_parent_pairs()

    def test_parent_selection_includes_elite_and_requires_distinct_scores(self):
        method, _, _ = make_method(pop_size=1)
        parent = make_function(1, score=1.0)
        elite = make_function(2, score=2.0)
        method._population.set_population([parent], increment_generation=False)
        method._elite_function = elite

        with patch("llm4ad.method.hsevo.hsevo.np.random.choice") as choice:
            choice.return_value = [elite, parent]
            pairs = method._select_parent_pairs()

        self.assertEqual(pairs, [[elite, parent]])
        pool = choice.call_args.args[0]
        self.assertEqual([func.score for func in pool], [2.0, 1.0])
        choice.assert_called_once()

    def test_generation_updates_good_and_bad_reflection_memory(self):
        reflection = "**Analysis:** better prefers small constants\n**Experience:** keep it simple"
        method, _, _ = make_method(
            responses=[reflection, "comprehensive", make_code(10)],
            scores=[10.0],
            pop_size=1,
            init_pop_size=0,
            mutation_rate=0.0,
            hs_attempts_per_generation=0,
        )
        parent = make_function(1, score=1.0)
        elite = make_function(2, score=2.0)
        method._population.set_population([parent], increment_generation=False)
        method._elite_function = elite

        with patch("llm4ad.method.hsevo.hsevo.np.random.choice") as choice:
            choice.return_value = [elite, parent]
            method._run_evolution_generation()

        self.assertEqual(method._good_reflections, ["keep it simple"])
        self.assertEqual(method._bad_reflections, [])
        self.assertEqual(method._elite_function.score, 10.0)

        method, _, _ = make_method(
            responses=[reflection, "comprehensive", make_code(1)],
            scores=[1.5],
            pop_size=1,
            init_pop_size=0,
            mutation_rate=0.0,
            hs_attempts_per_generation=0,
        )
        method._population.set_population([parent], increment_generation=False)
        method._elite_function = elite
        with patch("llm4ad.method.hsevo.hsevo.np.random.choice") as choice:
            choice.return_value = [elite, parent]
            method._run_evolution_generation()

        self.assertEqual(method._good_reflections, [])
        self.assertEqual(method._bad_reflections, ["keep it simple"])

    def test_harmony_parser_builds_parameterized_function(self):
        response = """
```python
def heuristic_v2(x, weight=0.5):
    return int(x[0] + weight)
```
```python
parameter_ranges = {"weight": (0.0, 1.0)}
```
"""
        ranges, function_source = HSEvoSampler.parse_harmony_response(response)
        func = HSEvoSampler.function_with_harmony_values(function_source, {"weight": 0.75})

        self.assertEqual(ranges, {"weight": (0.0, 1.0)})
        self.assertIn("weight = 0.75", func.body)
        self.assertIn("return int(x[0] + weight)", func.body)

    def test_harmony_search_evaluates_harmony_memory_and_keeps_best_score(self):
        hs_response = """
```python
def heuristic_v2(x, weight=0.5):
    return int(x[0] + weight)
```
```python
parameter_ranges = {"weight": (1.0, 1.0)}
```
"""
        method, _, _ = make_method(
            responses=[hs_response],
            scores=[1.0, 3.0],
            pop_size=1,
            init_pop_size=0,
            hm_size=2,
            max_iter=0,
            hs_attempts_per_generation=1,
        )
        source = make_function(1, score=1.0)
        method._population.set_population([source], increment_generation=False)
        method._elite_function = source

        best = method._harmony_search()

        self.assertIsNotNone(best)
        self.assertEqual(best.score, 3.0)
        self.assertEqual(method._tot_sample_nums, 2)
        self.assertTrue(method._population.has_hs_tried(source))
        self.assertIn(3.0, [func.score for func in method._population.population])

    def test_harmony_failures_do_not_break_main_state(self):
        method, _, _ = make_method(
            responses=["no parameter ranges here"],
            scores=[],
            pop_size=1,
            init_pop_size=0,
            hs_attempts_per_generation=1,
        )
        source = make_function(1, score=1.0)
        method._population.set_population([source], increment_generation=False)
        method._elite_function = source

        best = method._harmony_search()

        self.assertIsNone(best)
        self.assertEqual(method._tot_sample_nums, 0)
        self.assertEqual([func.score for func in method._population.population], [1.0])

    def test_defaults_match_original_hsevo_config(self):
        llm = ScriptedLLM()
        evaluation = FakeEvaluation()
        method = HSEvo(llm=llm, evaluation=evaluation, profiler=None)
        method._evaluation_executor.shutdown(cancel_futures=True)

        self.assertEqual(method._max_sample_nums, 450)
        self.assertEqual(method._pop_size, 10)
        self.assertEqual(method._init_pop_size, 30)
        self.assertEqual(method._mutation_rate, 0.5)
        self.assertEqual(method._hm_size, 5)
        self.assertEqual(method._hmcr, 0.7)
        self.assertEqual(method._par, 0.5)
        self.assertEqual(method._bandwidth, 0.2)
        self.assertEqual(method._max_iter, 5)
        self.assertEqual(method._hs_attempts_per_generation, 3)

        paras = Path("llm4ad/method/hsevo/paras.yaml").read_text()
        self.assertIn("max_sample_nums: 450", paras)
        self.assertIn("hm_size: 5", paras)
        self.assertIn("hs_attempts_per_generation: 3", paras)


if __name__ == "__main__":
    unittest.main()

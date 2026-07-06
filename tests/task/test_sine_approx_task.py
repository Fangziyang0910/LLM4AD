from __future__ import annotations

import math

from llm4ad.task.science_discovery.sine_approx.evaluation import SineApproxEvaluation


def linear_approx(x: float) -> float:
    return x


def cubic_approx(x: float) -> float:
    return x - x ** 3 / 6.0


def direct_sin(x: float) -> float:
    return math.sin(x)


def bad_output(x: float) -> float:
    return float("inf")


def test_sine_approx_evaluates_fixed_points():
    evaluator = SineApproxEvaluation()

    score = evaluator.evaluate_program("def approximate(x): return x", linear_approx)

    assert isinstance(score, float)
    assert 0.0 < score < 1.0
    assert evaluator.dataset_metadata["n_points"] == 161


def test_sine_approx_rewards_better_approximation():
    evaluator = SineApproxEvaluation()

    linear_score = evaluator.evaluate_program("def approximate(x): return x", linear_approx)
    cubic_score = evaluator.evaluate_program("def approximate(x): return x - x ** 3 / 6.0", cubic_approx)

    assert cubic_score > linear_score


def test_sine_approx_rejects_direct_sine_source():
    evaluator = SineApproxEvaluation()

    assert evaluator.evaluate_program("import math\ndef approximate(x): return math.sin(x)", direct_sin) is None


def test_sine_approx_rejects_nonfinite_output():
    evaluator = SineApproxEvaluation()

    assert evaluator.evaluate_program("def approximate(x): return float('inf')", bad_output) is None

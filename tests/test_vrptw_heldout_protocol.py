from experiments.evaluate_best import (
    TASK_SPECS,
    _parse_vrptw_units,
    _vrptw_eval_kwargs,
)


def test_vrptw_heldout_defaults_cover_same_and_larger_scales() -> None:
    spec = TASK_SPECS["vrptw_construct"]

    assert spec["default_units"] == (50, 100, 200)
    assert spec["unit_key"](100) == "vrptw100"


def test_vrptw_units_parse_positive_problem_sizes() -> None:
    assert _parse_vrptw_units("50,100,200") == [50, 100, 200]


def test_vrptw_eval_kwargs_hold_test_split_fixed_across_scales() -> None:
    kwargs = _vrptw_eval_kwargs(problem_size=200, timeout=120)

    assert kwargs == {
        "timeout_seconds": 120,
        "problem_size": 200,
        "n_instance": 16,
        "seed": 2025,
    }

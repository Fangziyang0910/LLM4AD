"""Regression tests for function-spec fixes and score-direction bugs."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from llm4ad.base.code import TextFunctionProgramConverter
from llm4ad.task.optimization.bp_2d_construct import template as bp2d_t
from llm4ad.task.optimization.co_bench.maximal_independent_set_co_bench import (
    template as mis_t,
)
from llm4ad.task.optimization.jssp_construct import template as jssp_t
from llm4ad.task.optimization.pymoo_moead import template as moead_t
from llm4ad.task.science_discovery.feynman_srsd import template as feynman_t


def _load_func(template_program: str):
    program = TextFunctionProgramConverter.text_to_program(template_program)
    assert program is not None and len(program.functions) == 1
    ns: dict = {}
    exec(template_program, ns, ns)
    return ns[program.functions[0].name]


def test_feynman_template_uses_xs_not_undefined_x():
    body = feynman_t.template_program
    assert "xs" in body
    # Avoid matching xs[ as x[
    tree = ast.parse(body)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "xs" in names
    assert "x" not in names
    fn = _load_func(body)
    xs = np.array([[1.0, 2.0], [3.0, 4.0]])
    params = np.ones(5)
    out = fn(xs, params)
    assert np.asarray(out).shape[0] == 2


def test_mis_template_is_executable_placeholder():
    fn = _load_func(mis_t.template_program)
    import networkx as nx

    g = nx.path_graph(3)
    sol = fn(g)
    assert isinstance(sol, dict) and "mis_nodes" in sol
    assert sol["mis_nodes"] == []


def test_jssp_defines_current_status_type():
    assert "class CurrentStatus" in jssp_t.template_program
    fn = _load_func(jssp_t.template_program)
    op = fn(
        {"machine_status": [0, 0], "job_status": [0, 0]},
        [(0, 0, 3), (1, 1, 1)],
    )
    assert op == (1, 1, 1)


def test_bp2d_description_mentions_occupancy_grid_not_corners_only():
    td = bp2d_t.task_description.lower()
    assert "occupancy" in td or "point_matrices" in td or "grid" in td
    assert "feasible corners" not in td


def test_bp1d_description_says_all_bin_capacities_are_passed():
    from llm4ad.task.optimization.bp_1d_construct import template as bp1d_t

    spec = bp1d_t.template_program.lower()
    assert "every available bin" in spec
    assert "capacities of feasible bins" not in spec


def test_moead_description_matches_higher_hv_search_score():
    td = moead_t.task_description.lower()
    assert "hypervolume" in td
    assert "higher" in td and "better" in td
    assert "lower" in td  # subproblem scalar is lower-better


@pytest.mark.parametrize(
    ("module_path", "expect_positive_better_ratio"),
    [
        (
            "llm4ad.task.optimization.co_bench.open_shop_scheduling_co_bench.evaluation",
            True,
        ),
        (
            "llm4ad.task.optimization.co_bench.p_median_capacitated_co_bench.evaluation",
            True,
        ),
        (
            "llm4ad.task.optimization.co_bench.generalised_assignment_problem_co_bench.evaluation",
            True,
        ),
    ],
)
def test_cobench_ratio_or_max_tasks_do_not_negate_higher_better_scores(
    module_path, expect_positive_better_ratio
):
    src = Path(*module_path.split(".")).with_suffix(".py").read_text(encoding="utf-8")
    # locate evaluate() return
    assert "return -np.mean(fitness_list)" not in src
    if expect_positive_better_ratio:
        assert "np.mean(fitness_list)" in src


def test_pymoo_moead_returns_positive_hypervolume():
    src = Path("llm4ad/task/optimization/pymoo_moead/evaluation.py").read_text(
        encoding="utf-8"
    )
    assert "return float(hv_value)" in src
    assert "return -hv_value" not in src


def test_no_template_body_uses_undefined_kwargs():
    root = Path("llm4ad/task")
    offenders = []
    for template_path in root.rglob("template.py"):
        tree = ast.parse(template_path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "template_program":
                    value = node.value
                    if isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute):
                        prog = ast.literal_eval(value.func.value)
                    else:
                        prog = ast.literal_eval(value)
                    ptree = ast.parse(prog)
                    funcs = [n for n in ptree.body if isinstance(n, ast.FunctionDef)]
                    assert len(funcs) == 1
                    func = funcs[0]
                    if func.args.kwarg is not None:
                        continue
                    body = ast.unparse(func)
                    if "kwargs[" in body or "kwargs.get" in body:
                        offenders.append(str(template_path))
    assert offenders == []


def test_tsp_description_does_not_claim_coordinate_inputs():
    from llm4ad.task.optimization.tsp_construct import template as tsp_t

    assert "does not receive coordinates" in tsp_t.task_description.lower()
    assert "distance matrix" in tsp_t.task_description.lower()


def test_cvrp_and_ovrp_descriptions_distinguish_closed_vs_open():
    from llm4ad.task.optimization.cvrp_construct import template as cvrp_t
    from llm4ad.task.optimization.ovrp_construct import template as ovrp_t

    assert "end at the depot" in cvrp_t.task_description.lower() or "start and end" in cvrp_t.task_description.lower()
    assert "does not charge the return" in ovrp_t.task_description.lower()
    assert cvrp_t.task_description.strip() != ovrp_t.task_description.strip()


def test_period_routing_evaluator_uses_customers_key():
    eval_src = Path(
        "llm4ad/task/optimization/co_bench/vehicle_routing_period_routing_co_bench/evaluation.py"
    ).read_text(encoding="utf-8")
    assert "j['customers']" in eval_src
    assert "costumers" not in eval_src


def test_online_bin_packing_2o_uses_own_template():
    eval_src = Path(
        "llm4ad/task/optimization/online_bin_packing_2O/evaluation.py"
    ).read_text(encoding="utf-8")
    assert "online_bin_packing_2O.template" in eval_src


def test_ode_1d_template_uses_solve_ivp_state_shape():
    from llm4ad.task.science_discovery.ode_1d import template as ode_t

    fn = _load_func(ode_t.template_program)
    out = np.asarray(fn(np.array([1.0]), np.ones(10)))
    assert out.shape == (1,)
    assert "length-one" in ode_t.template_program.lower()


def test_period_routing_description_allows_unused_vehicles():
    from llm4ad.task.optimization.co_bench.vehicle_routing_period_routing_co_bench import (
        template as pvrp_t,
    )

    spec = pvrp_t.template_program.lower()
    assert "at most" in spec
    assert "exactly equal to vehicles_per_day" not in spec


@pytest.mark.parametrize(
    ("filename", "expected_type", "expected_score"),
    [
        ("gap1.txt", "max", 5.0),
        ("gapa.txt", "min", -5.0),
    ],
)
def test_gap_uses_filename_objective_and_returns_higher_better_score(
    filename, expected_type, expected_score
):
    from llm4ad.task.optimization.co_bench.generalised_assignment_problem_co_bench.evaluation import (
        GAPEvaluationCB,
    )

    evaluation = GAPEvaluationCB.__new__(GAPEvaluationCB)
    evaluation._datasets = {filename: "1 1 1 5 1 10"}
    seen_types = []

    def solve(m, n, costs, consumption, capacities, problem_type):
        seen_types.append(problem_type)
        return {"assignments": [1]}

    assert evaluation.evaluate(solve) == expected_score
    assert seen_types == [expected_type]


def test_vrptw_problem_size_counts_customers_excluding_depot():
    from llm4ad.task.optimization.vrptw_construct.evaluation import VRPTWEvaluation

    evaluation = VRPTWEvaluation(problem_size=3, n_instance=1)
    coordinates, distances, demands, _, service, windows = evaluation._datasets[0]
    assert coordinates.shape == (4, 2)
    assert distances.shape == (4, 4)
    assert demands.shape == service.shape == (4,)
    assert windows.shape == (4, 2)


def test_vrptw_only_offers_feasible_customers_to_heuristic():
    from llm4ad.task.optimization.vrptw_construct.evaluation import VRPTWEvaluation

    evaluation = VRPTWEvaluation.__new__(VRPTWEvaluation)
    evaluation.problem_size = 2
    evaluation.n_instance = 1
    distances = np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    demands = np.array([0.0, 1.0, 10.0])
    service = np.zeros(3)
    windows = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    evaluation._datasets = [
        (np.zeros((3, 2)), distances, demands, 5.0, service, windows)
    ]
    offered = []

    def choose_first(
        current_node,
        depot,
        feasible_nodes,
        rest_capacity,
        current_time,
        demands_arg,
        distances_arg,
        windows_arg,
    ):
        offered.append(feasible_nodes.copy())
        return int(feasible_nodes[0])

    assert evaluation.evaluate(choose_first) is None
    assert offered
    assert all(2 not in nodes for nodes in offered)


def test_vrptw_cost_includes_final_return_to_depot():
    from llm4ad.task.optimization.vrptw_construct.evaluation import VRPTWEvaluation

    evaluation = VRPTWEvaluation.__new__(VRPTWEvaluation)
    evaluation.problem_size = 1
    evaluation.n_instance = 1
    distances = np.array([[0.0, 2.0], [2.0, 0.0]])
    evaluation._datasets = [
        (
            np.zeros((2, 2)),
            distances,
            np.array([0.0, 1.0]),
            5.0,
            np.zeros(2),
            np.array([[0.0, 10.0], [0.0, 10.0]]),
        )
    ]

    def choose_only_customer(*args):
        return int(args[2][0])

    assert evaluation.evaluate(choose_only_customer) == -4.0

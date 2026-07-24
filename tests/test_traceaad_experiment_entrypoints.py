from importlib import import_module
from pathlib import Path

from llm4ad.method.traceaad import TraceAAD


ENTRYPOINTS = (
    "experiments.tsp_construct.traceaad.run_experiment",
    "experiments.cvrp_aco.traceaad.run_experiment",
    "experiments.op_aco.traceaad.run_experiment",
    "experiments.online_bin_packing.traceaad.run_experiment",
    "experiments.knapsack_construct.traceaad.run_experiment",
    "experiments.tsp_gls.traceaad.run_experiment",
)


def test_traceaad_has_no_method_specific_experiment_runner() -> None:
    assert not Path("experiments/traceaad_runner.py").exists()


def test_each_task_entrypoint_builds_traceaad_directly(tmp_path: Path) -> None:
    for module_name in ENTRYPOINTS:
        module = import_module(module_name)
        method = module.build_method(log_dir=tmp_path / module_name)

        assert isinstance(method, TraceAAD)
        assert callable(module.main)

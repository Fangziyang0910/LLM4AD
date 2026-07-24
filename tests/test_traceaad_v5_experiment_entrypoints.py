from __future__ import annotations

from importlib import import_module
from pathlib import Path

from llm4ad.method.traceaad_v5 import TraceAADV5


ENTRYPOINTS = (
    "experiments.tsp_construct.traceaad_v5.run_experiment",
    "experiments.cvrp_aco.traceaad_v5.run_experiment",
    "experiments.op_aco.traceaad_v5.run_experiment",
    "experiments.online_bin_packing.traceaad_v5.run_experiment",
    "experiments.knapsack_construct.traceaad_v5.run_experiment",
    "experiments.tsp_gls.traceaad_v5.run_experiment",
)


def test_each_supported_task_builds_the_independent_v5_method(
    tmp_path: Path,
) -> None:
    for module_name in ENTRYPOINTS:
        module = import_module(module_name)
        method = module.build_method(log_dir=tmp_path / module_name)

        assert isinstance(method, TraceAADV5)
        assert callable(module.main)

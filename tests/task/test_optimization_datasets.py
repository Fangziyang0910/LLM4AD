from __future__ import annotations

from pathlib import Path

import pytest


OPTIMIZATION_TASKS = {
    "admissible_set",
    "bp_1d_construct",
    "bp_2d_construct",
    "bpp_offline_aco",
    "cflp_construct",
    "co_bench",
    "cvrp_aco",
    "cvrp_construct",
    "fssp_gls",
    "jssp_construct",
    "knapsack_construct",
    "max_cut",
    "online_bin_packing",
    "online_bin_packing_2O",
    "orienteering_construct",
    "ovrp_construct",
    "pymoo_moead",
    "qap_construct",
    "set_cover_construct",
    "tsp_aco",
    "tsp_construct",
    "tsp_gls_2O",
    "vrptw_construct",
}


@pytest.mark.parametrize("task", sorted(OPTIMIZATION_TASKS))
def test_optimization_task_has_dataset_protocol(task):
    task_dir = Path("llm4ad/task/optimization") / task
    assert (task_dir / "dataset.py").exists()
    assert (task_dir / "generate_dataset.py").exists()


def test_all_optimization_task_directories_are_covered():
    task_root = Path("llm4ad/task/optimization")
    task_dirs = {
        path.name
        for path in task_root.iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    }
    assert task_dirs == OPTIMIZATION_TASKS


def test_default_dataset_manifests_exist_for_generated_tasks():
    for task in sorted(OPTIMIZATION_TASKS - {"co_bench", "online_bin_packing_2O"}):
        manifest_path = Path("llm4ad/task/optimization") / task / "data" / "manifest.json"
        assert manifest_path.exists(), task

    assert Path("llm4ad/task/optimization/co_bench/dataset_manifest.json").exists()
    assert Path("llm4ad/task/optimization/online_bin_packing/data/manifest.json").exists()


def test_default_evaluators_use_train_split():
    from llm4ad.task.optimization.admissible_set.evaluation import ASPEvaluation
    from llm4ad.task.optimization.bp_1d_construct.evaluation import BP1DEvaluation
    from llm4ad.task.optimization.bp_2d_construct.evaluation import BP2DEvaluation
    from llm4ad.task.optimization.bpp_offline_aco.evaluation import BPPOfflineACOEvaluation
    from llm4ad.task.optimization.cflp_construct.evaluation import CFLPEvaluation
    from llm4ad.task.optimization.cvrp_aco.evaluation import CVRPACOEvaluation
    from llm4ad.task.optimization.cvrp_construct.evaluation import CVRPEvaluation
    from llm4ad.task.optimization.fssp_gls.evaluation import FSSPGLSEvaluation
    from llm4ad.task.optimization.jssp_construct.evaluation import JSSPEvaluation
    from llm4ad.task.optimization.knapsack_construct.evaluation import KnapsackEvaluation
    from llm4ad.task.optimization.max_cut.evaluation import MaxCutEvaluation
    from llm4ad.task.optimization.online_bin_packing.evaluation import OBPEvaluation
    from llm4ad.task.optimization.online_bin_packing_2O.evaluation import OBP_2O_Evaluation
    from llm4ad.task.optimization.orienteering_construct.evaluation import OrienteeringEvaluation
    from llm4ad.task.optimization.ovrp_construct.evaluation import OVRPEvaluation
    from llm4ad.task.optimization.pymoo_moead.evaluation import MOEAD_PYMOO_Evaluation
    from llm4ad.task.optimization.qap_construct.evaluation import QAPEvaluation
    from llm4ad.task.optimization.set_cover_construct.evaluation import SCPEvaluation
    from llm4ad.task.optimization.tsp_aco.evaluation import TSPACOEvaluation
    from llm4ad.task.optimization.tsp_construct.evaluation import TSPEvaluation
    from llm4ad.task.optimization.tsp_gls_2O.evaluation import TSP_GLS_2O_Evaluation
    from llm4ad.task.optimization.vrptw_construct.evaluation import VRPTWEvaluation

    evaluators = [
        ASPEvaluation(),
        BP1DEvaluation(),
        BP2DEvaluation(),
        BPPOfflineACOEvaluation(),
        CFLPEvaluation(),
        CVRPACOEvaluation(),
        CVRPEvaluation(),
        FSSPGLSEvaluation(),
        JSSPEvaluation(),
        KnapsackEvaluation(),
        MaxCutEvaluation(),
        OBPEvaluation(),
        OBP_2O_Evaluation(),
        OrienteeringEvaluation(),
        OVRPEvaluation(),
        MOEAD_PYMOO_Evaluation(),
        QAPEvaluation(),
        SCPEvaluation(),
        TSPACOEvaluation(),
        TSPEvaluation(),
        TSP_GLS_2O_Evaluation(),
        VRPTWEvaluation(),
    ]

    for evaluator in evaluators:
        assert evaluator.dataset_metadata["split"] == "train"

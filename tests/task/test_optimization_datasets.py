from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest


MAIN_OPTIMIZATION_TASKS = {
    "bo_acquisition",
    "bp_online",
    "bpp_offline_aco",
    "circle_packing",
    "cvrp_aco",
    "dpp_ga",
    "fssp_gls",
    "knapsack_construct",
    "mkp_aco",
    "online_bin_packing",
    "online_bin_packing_2O",
    "op_aco",
    "tsp_aco",
    "tsp_construct",
    "tsp_gls",
    "tsp_gls_2O",
}

OTHER_OPTIMIZATION_TASKS = {
    "aco_pheromone",
    "admissible_set",
    "bbob_metaheuristic",
    "bp_1d_construct",
    "bp_2d_construct",
    "cflp_construct",
    "cmaes_cov_update",
    "cvrp_construct",
    "deap_eaSimple_selection",
    "de_crossover_100d",
    "de_mutation",
    "es_step_size",
    "evo_dynamic",
    "gnn_aggregation",
    "jssp_construct",
    "large_scale_es",
    "max_cut",
    "moead_decomposition",
    "mobbob_metaheuristic",
    "nurse_rostering",
    "nsga2_crowding",
    "nsga2_pymoo",
    "one_plus_one",
    "orienteering_construct",
    "ovrp_construct",
    "portfolio_construct",
    "pymoo_moead",
    "pso_velocity",
    "qap_construct",
    "sa_acceptance",
    "set_cover_construct",
    "tabu_tsp",
    "tsp_rnr",
    "tpe_bandwidth",
    "vrptw_construct",
}

COBENCH_TASK = "cobench"
OPTIMIZATION_TASKS = MAIN_OPTIMIZATION_TASKS | OTHER_OPTIMIZATION_TASKS | {COBENCH_TASK}


def task_dir(task: str) -> Path:
    if task in MAIN_OPTIMIZATION_TASKS:
        return Path("llm4ad/task/optimization/main") / task
    if task in OTHER_OPTIMIZATION_TASKS:
        return Path("llm4ad/task/optimization/other") / task
    if task == COBENCH_TASK:
        return Path("llm4ad/task/optimization/cobench")
    raise AssertionError(f"unknown optimization task: {task}")


@pytest.mark.parametrize("task", sorted(OPTIMIZATION_TASKS))
def test_optimization_task_has_dataset_protocol(task):
    root = task_dir(task)
    assert (root / "dataset.py").exists()
    assert (root / "generate_dataset.py").exists()


def test_all_optimization_task_directories_are_covered():
    task_root = Path("llm4ad/task/optimization")
    top_level_dirs = {
        path.name
        for path in task_root.iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    }
    assert top_level_dirs == {"main", "other", COBENCH_TASK}

    main_dirs = {
        path.name
        for path in (task_root / "main").iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    }
    other_dirs = {
        path.name
        for path in (task_root / "other").iterdir()
        if path.is_dir() and (path / "__init__.py").exists()
    }
    assert main_dirs == MAIN_OPTIMIZATION_TASKS
    assert other_dirs == OTHER_OPTIMIZATION_TASKS


def test_default_dataset_manifests_exist_for_generated_tasks():
    for task in sorted(OPTIMIZATION_TASKS - {COBENCH_TASK, "online_bin_packing_2O"}):
        manifest_path = task_dir(task) / "data" / "manifest.json"
        assert manifest_path.exists(), task

    assert Path("llm4ad/task/optimization/cobench/dataset_manifest.json").exists()
    assert Path("llm4ad/task/optimization/main/online_bin_packing/data/manifest.json").exists()


def test_dataset_io_helpers_write_and_load_npz_split(tmp_path):
    from llm4ad.task.optimization.dataset_io import (
        format_dataset_summary,
        get_split_info,
        load_npz_split,
        write_npz_splits,
    )

    split_specs = {
        "train": {
            "role": "train",
            "filename": "train.npz",
            "n_instances": 2,
            "seed": 7,
        }
    }

    def generate_split(split, spec):
        data = np.arange(spec["n_instances"] * 2).reshape(spec["n_instances"], 2)
        return {"points": data}, {"shape": list(data.shape)}

    manifest = write_npz_splits(
        data_dir=tmp_path,
        dataset_id="toy_v1",
        task="toy",
        version=1,
        description="Toy dataset for testing the shared dataset helpers.",
        split_specs=split_specs,
        generate_split=generate_split,
        generator="tests.generate_split",
    )

    assert get_split_info(manifest, "train")["format"] == "npz_compressed"
    arrays, metadata = load_npz_split(data_dir=tmp_path, split="train")
    assert arrays["points"].tolist() == [[0, 1], [2, 3]]
    assert metadata["dataset_id"] == "toy_v1"
    assert metadata["split"] == "train"
    assert metadata["shape"] == [2, 2]
    assert format_dataset_summary(manifest) == "Wrote toy_v1 with 1 splits."


def test_generate_dataset_scripts_use_common_cli():
    for task in sorted(OPTIMIZATION_TASKS):
        text = (task_dir(task) / "generate_dataset.py").read_text()
        assert "run_dataset_cli" in text, task


def test_cobench_evaluators_use_common_base():
    from llm4ad.task.optimization.cobench.base import COBenchEvaluation

    cobench_root = task_dir(COBENCH_TASK)
    for evaluation_path in sorted(cobench_root.glob("*_co_bench/evaluation.py")):
        module_name = ".".join(evaluation_path.with_suffix("").parts)
        module = importlib.import_module(module_name)
        exported = getattr(module, "__all__", [])
        assert exported, evaluation_path
        for class_name in exported:
            evaluator_class = getattr(module, class_name)
            assert issubclass(evaluator_class, COBenchEvaluation), class_name
        assert "def evaluate_program" not in evaluation_path.read_text()


def test_default_evaluators_use_train_split():
    from llm4ad.task.optimization.other.admissible_set.evaluation import ASPEvaluation
    from llm4ad.task.optimization.other.aco_pheromone.evaluation import ACOPheromoneEvaluation
    from llm4ad.task.optimization.other.bbob_metaheuristic.evaluation import BBOBMetaheuristicEvaluation
    from llm4ad.task.optimization.main.bo_acquisition.evaluation import BOAcquisitionEvaluation
    from llm4ad.task.optimization.main.bp_online.evaluation import BPOnlineEvaluation
    from llm4ad.task.optimization.other.bp_1d_construct.evaluation import BP1DEvaluation
    from llm4ad.task.optimization.other.bp_2d_construct.evaluation import BP2DEvaluation
    from llm4ad.task.optimization.main.bpp_offline_aco.evaluation import BPPOfflineACOEvaluation
    from llm4ad.task.optimization.other.cflp_construct.evaluation import CFLPEvaluation
    from llm4ad.task.optimization.main.circle_packing.evaluation import CirclePackingEvaluation
    from llm4ad.task.optimization.other.cmaes_cov_update.evaluation import CMAESCovUpdateEvaluation
    from llm4ad.task.optimization.main.cvrp_aco.evaluation import CVRPACOEvaluation
    from llm4ad.task.optimization.other.cvrp_construct.evaluation import CVRPEvaluation
    from llm4ad.task.optimization.other.deap_eaSimple_selection.evaluation import EASimpleSelectionEvaluation
    from llm4ad.task.optimization.other.de_crossover_100d.evaluation import DECrossover100DEvaluation
    from llm4ad.task.optimization.other.de_mutation.evaluation import DEMutationEvaluation
    from llm4ad.task.optimization.main.dpp_ga.evaluation import DPPGAEvaluation
    from llm4ad.task.optimization.other.es_step_size.evaluation import ESStepSizeEvaluation
    from llm4ad.task.optimization.other.evo_dynamic.evaluation import EvoDynamicEvaluation
    from llm4ad.task.optimization.main.fssp_gls.evaluation import FSSPGLSEvaluation
    from llm4ad.task.optimization.other.gnn_aggregation.evaluation import GNNAggregationEvaluation
    from llm4ad.task.optimization.other.jssp_construct.evaluation import JSSPEvaluation
    from llm4ad.task.optimization.main.knapsack_construct.evaluation import KnapsackEvaluation
    from llm4ad.task.optimization.other.large_scale_es.evaluation import LargeScaleESEvaluation
    from llm4ad.task.optimization.other.max_cut.evaluation import MaxCutEvaluation
    from llm4ad.task.optimization.main.mkp_aco.evaluation import MKPACOEvaluation
    from llm4ad.task.optimization.other.moead_decomposition.evaluation import MOEADDecompositionEvaluation
    from llm4ad.task.optimization.other.mobbob_metaheuristic.evaluation import MoBBOBMetaheuristicEvaluation
    from llm4ad.task.optimization.other.nurse_rostering.evaluation import NurseRosteringEvaluation
    from llm4ad.task.optimization.other.nsga2_crowding.evaluation import NSGA2CrowdingEvaluation
    from llm4ad.task.optimization.other.nsga2_pymoo.evaluation import NSGA2PymooEvaluation
    from llm4ad.task.optimization.main.online_bin_packing.evaluation import OBPEvaluation
    from llm4ad.task.optimization.main.online_bin_packing_2O.evaluation import OBP_2O_Evaluation
    from llm4ad.task.optimization.other.one_plus_one.evaluation import OnePlusOneEvaluation
    from llm4ad.task.optimization.main.op_aco.evaluation import OPACOEvaluation
    from llm4ad.task.optimization.other.orienteering_construct.evaluation import OrienteeringEvaluation
    from llm4ad.task.optimization.other.ovrp_construct.evaluation import OVRPEvaluation
    from llm4ad.task.optimization.other.portfolio_construct.evaluation import PortfolioConstructEvaluation
    from llm4ad.task.optimization.other.pymoo_moead.evaluation import MOEAD_PYMOO_Evaluation
    from llm4ad.task.optimization.other.pso_velocity.evaluation import PSOVelocityEvaluation
    from llm4ad.task.optimization.other.qap_construct.evaluation import QAPEvaluation
    from llm4ad.task.optimization.other.sa_acceptance.evaluation import SAAcceptanceEvaluation
    from llm4ad.task.optimization.other.set_cover_construct.evaluation import SCPEvaluation
    from llm4ad.task.optimization.other.tabu_tsp.evaluation import TabuTSPEvaluation
    from llm4ad.task.optimization.main.tsp_aco.evaluation import TSPACOEvaluation
    from llm4ad.task.optimization.main.tsp_construct.evaluation import TSPEvaluation
    from llm4ad.task.optimization.main.tsp_gls.evaluation import TSPGLSEvaluation
    from llm4ad.task.optimization.main.tsp_gls_2O.evaluation import TSP_GLS_2O_Evaluation
    from llm4ad.task.optimization.other.tsp_rnr.evaluation import TSPRnrEvaluation
    from llm4ad.task.optimization.other.tpe_bandwidth.evaluation import TPEBandwidthEvaluation
    from llm4ad.task.optimization.other.vrptw_construct.evaluation import VRPTWEvaluation

    evaluators = [
        ASPEvaluation(),
        ACOPheromoneEvaluation(n_ants=3, iter_max=2, n_runs=1),
        BBOBMetaheuristicEvaluation(budget=10, n_runs=1),
        BOAcquisitionEvaluation(),
        BPOnlineEvaluation(),
        BP1DEvaluation(),
        BP2DEvaluation(),
        BPPOfflineACOEvaluation(),
        CFLPEvaluation(),
        CirclePackingEvaluation(),
        CMAESCovUpdateEvaluation(max_evals=20, n_runs=1),
        CVRPACOEvaluation(),
        CVRPEvaluation(),
        EASimpleSelectionEvaluation(pop_size=8, n_gen=2, n_runs=1),
        DECrossover100DEvaluation(pop_size=5, max_evals=20, n_runs=1),
        DEMutationEvaluation(pop_size=5, max_evals=20, n_runs=1),
        DPPGAEvaluation(),
        ESStepSizeEvaluation(lam=3, max_evals=20, n_runs=1),
        EvoDynamicEvaluation(pop_size=8, k_iter=2),
        FSSPGLSEvaluation(),
        GNNAggregationEvaluation(),
        JSSPEvaluation(),
        KnapsackEvaluation(),
        LargeScaleESEvaluation(max_evals=20, n_runs=1),
        MaxCutEvaluation(),
        MKPACOEvaluation(),
        MOEADDecompositionEvaluation(n_gen=2, n_runs=1, hv_samples=500),
        MoBBOBMetaheuristicEvaluation(budget=10, n_runs=1),
        NurseRosteringEvaluation(),
        NSGA2CrowdingEvaluation(pop_size=10, n_gen=2, n_runs=1),
        NSGA2PymooEvaluation(pop_size=10, n_gen=2, n_runs=1),
        OBPEvaluation(),
        OBP_2O_Evaluation(),
        OnePlusOneEvaluation(max_evals=6, n_runs=1),
        OPACOEvaluation(),
        OrienteeringEvaluation(),
        OVRPEvaluation(),
        PortfolioConstructEvaluation(),
        MOEAD_PYMOO_Evaluation(),
        PSOVelocityEvaluation(pop_size=5, max_iterations=3, n_runs=1),
        QAPEvaluation(),
        SAAcceptanceEvaluation(max_iter=20, n_runs=1),
        SCPEvaluation(),
        TabuTSPEvaluation(n_iter=5, n_runs=1),
        TSPACOEvaluation(),
        TSPEvaluation(),
        TSPGLSEvaluation(),
        TSP_GLS_2O_Evaluation(),
        TSPRnrEvaluation(),
        TPEBandwidthEvaluation(n_startup=2, n_iter=3, n_runs=1),
        VRPTWEvaluation(),
    ]

    for evaluator in evaluators:
        assert evaluator.dataset_metadata["split"] == "train"

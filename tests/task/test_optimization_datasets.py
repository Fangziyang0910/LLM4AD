from __future__ import annotations

from pathlib import Path

import pytest


OPTIMIZATION_TASKS = {
    "admissible_set",
    "aco_pheromone",
    "bbob_metaheuristic",
    "bo_acquisition",
    "bp_online",
    "bp_1d_construct",
    "bp_2d_construct",
    "bpp_offline_aco",
    "cflp_construct",
    "circle_packing",
    "cmaes_cov_update",
    "co_bench",
    "cvrp_aco",
    "cvrp_construct",
    "deap_eaSimple_selection",
    "de_crossover_100d",
    "de_mutation",
    "dpp_ga",
    "es_step_size",
    "evo_dynamic",
    "fssp_gls",
    "gnn_aggregation",
    "jssp_construct",
    "knapsack_construct",
    "large_scale_es",
    "max_cut",
    "mkp_aco",
    "moead_decomposition",
    "mobbob_metaheuristic",
    "nurse_rostering",
    "nsga2_crowding",
    "nsga2_pymoo",
    "online_bin_packing",
    "online_bin_packing_2O",
    "op_aco",
    "orienteering_construct",
    "ovrp_construct",
    "portfolio_construct",
    "pymoo_moead",
    "pso_velocity",
    "qap_construct",
    "sa_acceptance",
    "set_cover_construct",
    "tabu_tsp",
    "tsp_aco",
    "tsp_construct",
    "tsp_gls",
    "tsp_gls_2O",
    "tsp_rnr",
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
    from llm4ad.task.optimization.aco_pheromone.evaluation import ACOPheromoneEvaluation
    from llm4ad.task.optimization.bbob_metaheuristic.evaluation import BBOBMetaheuristicEvaluation
    from llm4ad.task.optimization.bo_acquisition.evaluation import BOAcquisitionEvaluation
    from llm4ad.task.optimization.bp_online.evaluation import BPOnlineEvaluation
    from llm4ad.task.optimization.bp_1d_construct.evaluation import BP1DEvaluation
    from llm4ad.task.optimization.bp_2d_construct.evaluation import BP2DEvaluation
    from llm4ad.task.optimization.bpp_offline_aco.evaluation import BPPOfflineACOEvaluation
    from llm4ad.task.optimization.cflp_construct.evaluation import CFLPEvaluation
    from llm4ad.task.optimization.circle_packing.evaluation import CirclePackingEvaluation
    from llm4ad.task.optimization.cmaes_cov_update.evaluation import CMAESCovUpdateEvaluation
    from llm4ad.task.optimization.cvrp_aco.evaluation import CVRPACOEvaluation
    from llm4ad.task.optimization.cvrp_construct.evaluation import CVRPEvaluation
    from llm4ad.task.optimization.deap_eaSimple_selection.evaluation import EASimpleSelectionEvaluation
    from llm4ad.task.optimization.de_crossover_100d.evaluation import DECrossover100DEvaluation
    from llm4ad.task.optimization.de_mutation.evaluation import DEMutationEvaluation
    from llm4ad.task.optimization.dpp_ga.evaluation import DPPGAEvaluation
    from llm4ad.task.optimization.es_step_size.evaluation import ESStepSizeEvaluation
    from llm4ad.task.optimization.evo_dynamic.evaluation import EvoDynamicEvaluation
    from llm4ad.task.optimization.fssp_gls.evaluation import FSSPGLSEvaluation
    from llm4ad.task.optimization.gnn_aggregation.evaluation import GNNAggregationEvaluation
    from llm4ad.task.optimization.jssp_construct.evaluation import JSSPEvaluation
    from llm4ad.task.optimization.knapsack_construct.evaluation import KnapsackEvaluation
    from llm4ad.task.optimization.large_scale_es.evaluation import LargeScaleESEvaluation
    from llm4ad.task.optimization.max_cut.evaluation import MaxCutEvaluation
    from llm4ad.task.optimization.mkp_aco.evaluation import MKPACOEvaluation
    from llm4ad.task.optimization.moead_decomposition.evaluation import MOEADDecompositionEvaluation
    from llm4ad.task.optimization.mobbob_metaheuristic.evaluation import MoBBOBMetaheuristicEvaluation
    from llm4ad.task.optimization.nurse_rostering.evaluation import NurseRosteringEvaluation
    from llm4ad.task.optimization.nsga2_crowding.evaluation import NSGA2CrowdingEvaluation
    from llm4ad.task.optimization.nsga2_pymoo.evaluation import NSGA2PymooEvaluation
    from llm4ad.task.optimization.online_bin_packing.evaluation import OBPEvaluation
    from llm4ad.task.optimization.online_bin_packing_2O.evaluation import OBP_2O_Evaluation
    from llm4ad.task.optimization.op_aco.evaluation import OPACOEvaluation
    from llm4ad.task.optimization.orienteering_construct.evaluation import OrienteeringEvaluation
    from llm4ad.task.optimization.ovrp_construct.evaluation import OVRPEvaluation
    from llm4ad.task.optimization.portfolio_construct.evaluation import PortfolioConstructEvaluation
    from llm4ad.task.optimization.pymoo_moead.evaluation import MOEAD_PYMOO_Evaluation
    from llm4ad.task.optimization.pso_velocity.evaluation import PSOVelocityEvaluation
    from llm4ad.task.optimization.qap_construct.evaluation import QAPEvaluation
    from llm4ad.task.optimization.sa_acceptance.evaluation import SAAcceptanceEvaluation
    from llm4ad.task.optimization.set_cover_construct.evaluation import SCPEvaluation
    from llm4ad.task.optimization.tabu_tsp.evaluation import TabuTSPEvaluation
    from llm4ad.task.optimization.tsp_aco.evaluation import TSPACOEvaluation
    from llm4ad.task.optimization.tsp_construct.evaluation import TSPEvaluation
    from llm4ad.task.optimization.tsp_gls.evaluation import TSPGLSEvaluation
    from llm4ad.task.optimization.tsp_gls_2O.evaluation import TSP_GLS_2O_Evaluation
    from llm4ad.task.optimization.tsp_rnr.evaluation import TSPRnrEvaluation
    from llm4ad.task.optimization.vrptw_construct.evaluation import VRPTWEvaluation

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
        VRPTWEvaluation(),
    ]

    for evaluator in evaluators:
        assert evaluator.dataset_metadata["split"] == "train"

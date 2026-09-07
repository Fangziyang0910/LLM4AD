import numpy as np

from experiments.behavesim_population_calibration.run import (
    compare_matrices,
    landscape,
)
from experiments.traceaad_refine_e1 import profile_core as core
from llm4ad.task.optimization.generated_data_config import get_generated_task_kwargs
from llm4ad.task.optimization.online_bin_packing.evaluation import OBPEvaluation


def test_obp_matched_probe_uses_training_lengths_and_capacities():
    compact = core._build_probe_data("online_bin_packing", "A", obp_scale="compact")
    matched = core._build_probe_data("online_bin_packing", "A", obp_scale="matched")

    assert sorted(len(value["items"]) for value in compact["instances"]) == [256] * 4
    assert sorted(len(value["items"]) for value in matched["instances"]) == [1000, 1000, 5000, 5000]
    assert sorted(value["capacity"] for value in matched["instances"]) == [100, 100, 500, 500]
    assert matched["probe_metadata"]["scale"] == "matched"


def test_aco_probe_records_all_fixed_random_streams():
    offsets = core.CALIBRATED_ACO_SEED_OFFSETS
    data = core._build_probe_data("op_aco", "A", aco_seed_offsets=offsets)

    assert data["probe_metadata"]["aco_seed_offsets"] == list(offsets)
    assert data["probe_metadata"]["instances"] == 4
    assert data["probe_metadata"]["role"] == "training-probe"


def test_aco_multistream_offset_zero_reproduces_single_stream_trajectory():
    functions = {
        "op_aco": lambda prizes, distances, max_len: np.maximum(
            prizes[np.newaxis, :] / distances, 1e-9
        ),
        "cvrp_aco": lambda distances, coordinates, demands, capacity: 1.0
        / distances,
    }
    for task, function in functions.items():
        trajectories = {}
        for name, offsets in {
            "single": (0,),
            "multi": core.CALIBRATED_ACO_SEED_OFFSETS,
        }.items():
            core._init_worker(
                task,
                "A",
                core.DEFAULT_TRAJECTORY_POINTS[task],
                30.0,
                aco_seed_offsets=offsets,
            )
            evaluator = core._GLOBAL_DATA["evaluator"]
            evaluator.n_ants = 3
            evaluator.n_iterations = 2
            if task == "op_aco":
                trajectories[name], _ = core._profile_op_aco(function)
            else:
                trajectories[name], _ = core._profile_cvrp_aco(function)
        assert trajectories["multi"][::4] == trajectories["single"]


def test_obp_matched_profile_score_matches_official_evaluator():
    def priority(item, valid_bins):  # noqa: ARG001
        return -valid_bins

    core._init_worker(
        "online_bin_packing",
        "A",
        core.DEFAULT_TRAJECTORY_POINTS["online_bin_packing"],
        30.0,
        obp_scale=core.CALIBRATED_OBP_SCALE,
    )
    _, profile_score = core._profile_obp(priority)

    kwargs = get_generated_task_kwargs("online_bin_packing", "train")
    kwargs["seed"] = 42
    evaluator = OBPEvaluation(**kwargs)
    assert profile_score == evaluator.evaluate(priority)


def test_landscape_statistics_detect_ordered_population_geometry():
    positions = np.arange(8, dtype=float)
    matrix = np.abs(positions[:, None] - positions[None, :])
    result = landscape(matrix, positions)

    assert result["distance_fitness_gap_spearman"] > 0.98
    assert result["random_minus_knn_gap"] > 0
    assert result["far_minus_near_gap"] > 0
    comparison = compare_matrices(matrix, matrix, k=2)
    assert np.isclose(comparison["distance_spearman"], 1.0)
    assert comparison["knn_overlap"] == 1.0
